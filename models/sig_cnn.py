from dataclasses import dataclass, asdict
from enum import Enum, auto
from typing import Type, Union

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
)
from preprocessor_meta import filter_sig_meta


def to_odd(i: int) -> int:
    return (i // 2) * 2 + 1


@argsclass(name="sig_cnn")
@dataclass
class SigCnnHp(HpWithRegressionHead):
    batch: int = 512
    epochs: int = 1
    lr: float = 0.01
    dtype: torch.dtype = torch.float32

    linear1drop: float = 0.2
    linear1out: int = 64  # if this value is 1, then omit linear2
    head: RegressionHead = RegressionHead.LINEAR

    convdrop: float = 0.2

    conv1w: int = 105
    conv1out: int = 150
    conv1stride: int = 1
    mp1w: int = 1

    conv2w: int = 8
    conv2out: int = 40
    conv2stride: int = 1
    mp2w: int = 4

    conv3w: int = 8
    conv3out: int = 50
    conv3stride: int = 1
    mp3w: int = 1

    conv4w: int = 33
    conv4out: int = 50
    conv4stride: int = 1
    mp4w: int = 3

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


class Cnn(nn.Module):
    """
    Applies a CNN to qtransform data and produces an output shaped like
    (batch, channels, width). Dimension sizes depend on
    hyper-parameters.
    """

    def __init__(
        self, device: torch.device, hp: SigCnnHp, apply_final_activation: bool
    ):
        super().__init__()
        self.device = device
        self.hp = hp
        self.apply_final_activation = apply_final_activation

        self.conv1 = ConvBlock(
            w=hp.conv1w,
            in_channels=N_SIGNALS,
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
        self.conv_dropout = nn.Dropout(p=self.hp.convdrop)
        outw = (
            filter_sig_meta.output_shape[1]
            // hp.conv1stride
            // hp.mp1w
            // hp.conv2stride
            // hp.mp2w
            // hp.conv3stride
            // hp.mp3w
            // hp.conv4stride
            // hp.mp4w
        )
        if outw == 0:
            raise ValueError("strides and maxpools took output width to zero")
        self.output_shape = (hp.conv4out, outw)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x, use_activation=True)
        out = self.conv2(out, use_activation=True)
        out = self.conv3(out, use_activation=True)
        out = self.conv4(out, use_activation=self.apply_final_activation)
        return out


class Model(nn.Module):
    def __init__(self, cnn: nn.Module, head: nn.Module):
        super().__init__()
        self.cnn = cnn
        self.head = head

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.cnn(x))


class Manager(ModelManager):
    def train(
        self,
        data_dir: Path,
        n: Optional[int],
        device: torch.device,
        hp: HyperParameters,
    ):
        if not isinstance(hp, SigCnnHp):
            raise ValueError("wrong hyper-parameter class: {hp}")

        wandb.init(project="g2net-" + __name__, entity="wisdom", config=asdict(hp))

        head_class: Union[Type[MaxHead], Type[LinearHead]] = (
            MaxHead if hp.head == RegressionHead.MAX else LinearHead
        )
        cnn = Cnn(device, hp, head_class.apply_activation_before_input)
        head = head_class(device, hp, cnn.output_shape)
        model = Model(cnn, head)

        self._train(model, device, data_dir, n, [filter_sig_meta], hp)
