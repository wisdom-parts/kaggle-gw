import math
from abc import ABC
from dataclasses import dataclass, asdict
from enum import Enum, auto
from typing import Type, Union, Dict, Tuple, cast

import numpy as np
import torch
import wandb
from datargs import argsclass
from nnAudio.Spectrogram import CQT2010v2
from torch import nn, Tensor

from gw_data import *
from models import (
    HyperParameters,
    ModelManager,
    MaxHead,
    LinearHead,
    HpWithRegressionHead,
    RegressionHead,
    to_odd,
)
from preprocessor_meta import Preprocessor, PreprocessorMeta


class CqtInputLayer(nn.Module):
    out_w = 256
    hop_len = 4096 // out_w
    sampling_rate = SIGNAL_LEN / SIGNAL_SECS

    # CQT2010v2 doesn't respect fmax. It ends up adjusting the frequency end-points and then
    # failing because its new fmax exceeds the Nyquist frequency. So we have to compute the
    # octaves and bins ourselves.
    fmin = 32.0
    fmax = 512.0
    bins_per_octave = 12
    n_octaves = round(math.log(fmax / fmin, 2))
    n_bins = n_octaves * bins_per_octave

    def __init__(self, in_shape: Tuple[int, ...]):
        super().__init__()
        if in_shape != (N_SIGNALS, SIGNAL_LEN):
            raise ValueError(
                f"this layer only supports raw signals; got shape {in_shape}"
            )
        self.cqt = CQT2010v2(
            sr=self.sampling_rate,
            hop_length=self.hop_len,
            fmin=self.fmin,
            n_bins=self.n_bins,
            bins_per_octave=self.bins_per_octave,
            norm=False,
            output_format="Magnitude",  # TODO: "Complex"
        )
        self.freqs = self.cqt.frequencies
        self.times = np.array(
            [(i / self.out_w) * SIGNAL_SECS for i in range(self.out_w)]
        )
        self.output_shape = (N_SIGNALS, len(self.freqs), self.out_w)
        self.bn = nn.BatchNorm2d(N_SIGNALS)

    def forward(self, x: Tensor) -> Tensor:
        # CQT2010v2 doesn't support separate batch and channel dimensions, so combine them.
        out = self.cqt(torch.flatten(x, start_dim=0, end_dim=1))
        # CQT2010v2 returns a value for each time endpoint. We'd prefer exactly out_w values.
        out = out[:, :, 0:-1]
        # Separate the batch and channel dimensions again.
        out = out.view(x.shape[0], *self.output_shape)
        out = self.bn(out)
        return out


class InputLayer(Enum):
    CQT = auto()


@argsclass(name="cnn1d")
@dataclass
class Cnn1dHp(HpWithRegressionHead):
    batch: int = 600
    epochs: int = 1
    lr: float = 0.01
    dtype: torch.dtype = torch.float32

    linear1drop: float = 0.0
    linear1out: int = 64  # if this value is 1, then omit linear2
    head: RegressionHead = RegressionHead.LINEAR

    preprocessor: Preprocessor = Preprocessor.QTRANSFORM

    inputlayer: Optional[InputLayer] = None

    conv1w: int = 5
    conv1out: int = 200
    conv1stride: int = 1
    mp1w: int = 2

    conv2w: int = 17
    conv2out: int = 130
    conv2stride: int = 2
    mp2w: int = 2

    conv3w: int = 17
    conv3out: int = 65
    conv3stride: int = 2
    mp3w: int = 1

    conv4w: int = 7
    conv4out: int = 180
    conv4stride: int = 2
    mp4w: int = 1

    @property
    def manager_class(self) -> Type[ModelManager]:
        return Manager

    def __post_init__(self):
        self.conv1w = to_odd(self.conv1w)
        self.conv2w = to_odd(self.conv2w)
        self.conv3w = to_odd(self.conv3w)
        self.conv4w = to_odd(self.conv4w)


class ConvBlock(nn.Module):
    def __init__(
        self, w: int, in_channels: int, out_channels: int, stride: int, mpw: int
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(w,),
            stride=(stride,),
            padding=w // 2,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()
        self.mp = nn.MaxPool1d(kernel_size=mpw)

    def forward(self, x: Tensor, use_activation: bool):
        out = self.conv(x)
        out = self.bn(out)
        if use_activation:
            out = self.activation(out)
        out = self.mp(out)
        return out


class Cnn1d(nn.Module):
    """
    Takes an input of at least two dimensions, with the last treated as in_width and the rest
    treated as channels. Applies a 1D cnn to product an output shaped (out_channels, out_width),
    where the sizes depend on hyper-parameters. (Additionally, there's a batch dimension at the
    front of every shape.)
    """

    def __init__(self, hp: Cnn1dHp, apply_final_activation: bool):
        super().__init__()
        self.hp = hp
        self.apply_final_activation = apply_final_activation

        preprocessor_meta: PreprocessorMeta = hp.preprocessor.value
        in_shape = preprocessor_meta.output_shape

        self.input_layer = None
        if hp.inputlayer:
            input_layer_class = CqtInputLayer
            self.input_layer = input_layer_class(in_shape)
            conv_in_shape = self.input_layer.output_shape
        else:
            conv_in_shape = cast(Tuple[int, int, int], in_shape)

        if len(in_shape) < 2:
            raise ValueError(
                f"Cnn1d requires at least 2 dimensions; got {len(in_shape)}"
            )
        conv_in_channels = int(np.prod(conv_in_shape[0:-1]))
        conv_in_w = conv_in_shape[-1]

        # includes batch dimension
        self.last_in_channel_dim = len(conv_in_shape) - 1

        self.conv1 = ConvBlock(
            w=hp.conv1w,
            in_channels=conv_in_channels,
            out_channels=hp.conv1out,
            stride=hp.conv1stride,
            mpw=hp.mp1w,
        )
        self.conv2 = ConvBlock(
            w=hp.conv2w,
            in_channels=hp.conv1out,
            out_channels=hp.conv2out,
            stride=hp.conv2stride,
            mpw=hp.mp2w,
        )
        self.conv3 = ConvBlock(
            w=hp.conv3w,
            in_channels=hp.conv2out,
            out_channels=hp.conv3out,
            stride=hp.conv3stride,
            mpw=hp.mp3w,
        )
        self.conv4 = ConvBlock(
            w=hp.conv4w,
            in_channels=hp.conv3out,
            out_channels=hp.conv4out,
            stride=hp.conv4stride,
            mpw=hp.mp4w,
        )
        out_w = (
            conv_in_w
            // hp.conv1stride
            // hp.mp1w
            // hp.conv2stride
            // hp.mp2w
            // hp.conv3stride
            // hp.mp3w
            // hp.conv4stride
            // hp.mp4w
        )
        wandb.log({"conv_out_width": out_w})
        if out_w == 0:
            raise ValueError("strides and maxpools took output width to zero")
        self.output_shape = (hp.conv4out, out_w)

    def forward(self, x: Tensor) -> Tensor:
        out = self.input_layer(x) if self.input_layer else x

        # Flatten to a single channel dimension.
        out = torch.flatten(out, 1, self.last_in_channel_dim)

        out = self.conv1(out, use_activation=True)
        out = self.conv2(out, use_activation=True)
        out = self.conv3(out, use_activation=True)
        out = self.conv4(out, use_activation=self.apply_final_activation)
        return out


class Model(nn.Module):
    def __init__(self, cnn: nn.Module, head: nn.Module, hp: Cnn1dHp):
        super().__init__()
        self.cnn = cnn
        self.head = head
        self.hp = hp

    def forward(self, xd: Dict[str, Tensor]) -> Tensor:
        return self.head(self.cnn(xd[self.hp.preprocessor.value.name]))


class Manager(ModelManager):
    def train(
        self,
        data_dir: Path,
        n: Optional[int],
        device: torch.device,
        hp: HyperParameters,
    ):
        if not isinstance(hp, Cnn1dHp):
            raise ValueError("wrong hyper-parameter class: {hp}")

        wandb.init(project="g2net-" + __name__, entity="wisdom", config=asdict(hp))

        head_class: Union[Type[MaxHead], Type[LinearHead]] = (
            MaxHead if hp.head == RegressionHead.MAX else LinearHead
        )
        cnn = Cnn1d(hp, head_class.apply_activation_before_input)
        head = head_class(hp, cnn.output_shape)
        model = Model(cnn, head, hp)

        self._train(model, device, data_dir, n, [hp.preprocessor.value], hp)
