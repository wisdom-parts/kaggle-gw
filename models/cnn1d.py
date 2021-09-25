from dataclasses import dataclass, asdict
from typing import Type, Union, Dict

import torch
import wandb
from datargs import argsclass
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


@argsclass(name="cnn1d")
@dataclass
class Cnn1dHp(HpWithRegressionHead):
    batch: int = 250
    epochs: int = 1
    lr: float = 0.001
    dtype: torch.dtype = torch.float32

    linear1drop: float = 0.0
    linear1out: int = 64  # if this value is 1, then omit linear2
    head: RegressionHead = RegressionHead.LINEAR

    preprocessor: Preprocessor = Preprocessor.FILTER_SIG

    conv1w: int = 111
    conv1out: int = 100
    conv1stride: int = 1
    mp1w: int = 2

    conv2w: int = 7
    conv2out: int = 50
    conv2stride: int = 2
    mp2w: int = 3

    conv3w: int = 15
    conv3out: int = 70
    conv3stride: int = 2
    mp3w: int = 4

    conv4w: int = 43
    conv4out: int = 70
    conv4stride: int = 2
    mp4w: int = 2

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
        if len(in_shape) < 2:
            raise ValueError(
                f"Cnn1d requires at least 2 dimensions; got {len(in_shape)}"
            )
        in_channels = int(np.prod(in_shape[0:-1]))
        in_w = in_shape[-1]

        # includes batch dimension
        self.last_in_channel_dim = len(in_shape) - 1

        self.conv1 = ConvBlock(
            w=hp.conv1w,
            in_channels=in_channels,
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
            in_w
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

        # Flatten to a single channel dimension.
        out = torch.flatten(x, 1, self.last_in_channel_dim)

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
