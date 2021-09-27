from dataclasses import dataclass, asdict
from enum import Enum, auto
from typing import Type

import torch
import wandb
from datargs import argsclass
from torch import nn, Tensor

from preprocessor_meta import Preprocessor, filter_sig_meta
from gw_data import *
from models import HyperParameters, ModelManager


class RegressionHead(Enum):
    LINEAR = auto()
    AVG_LINEAR = auto()


def to_odd(i: int) -> int:
    return (i // 2) * 2 + 1


@argsclass(name="sig_cnn")
@dataclass
class SigCnnHp(HyperParameters):
    batch: int = 512
    epochs: int = 10
    lr: float = 0.0003
    dtype: torch.dtype = torch.float32

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

    head: RegressionHead = RegressionHead.AVG_LINEAR

    lindrop: float = 0.22

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


class SigCnn(nn.Module):
    """
    Applies a CNN to the output of preprocess filter_sig and produces one logit as output.
    input size: (batch_size, N_SIGNALS, SIGNAL_LEN)
    output size: (batch_size, 1)
    """

    def __init__(self, device: torch.device, hp: SigCnnHp):
        super().__init__()
        self.hp = hp
        self.device = device

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
        self.outw = (
            SIGNAL_LEN
            // hp.conv1stride
            // hp.mp1w
            // hp.conv2stride
            // hp.mp2w
            // hp.conv3stride
            // hp.mp3w
            // hp.conv4stride
            // hp.mp4w
        )
        if self.outw == 0:
            raise ValueError("strides and maxpools took output width to zero")
        self.linear_dropout = nn.Dropout(hp.lindrop)
        linear_in_features = hp.conv4out * (
            self.outw if hp.head == RegressionHead.LINEAR else 1
        )
        self.linear = nn.Linear(in_features=linear_in_features, out_features=1)

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.size()[0]

        out = self.conv1(x, use_activation=True)
        out = self.conv2(out, use_activation=True)
        out = self.conv3(out, use_activation=True)
        out = self.conv4(out, use_activation=False)

        if self.hp.head == RegressionHead.LINEAR:
            out = torch.flatten(out, start_dim=1)
        else:
            assert self.hp.head == RegressionHead.AVG_LINEAR
            # Average across w, leaving (batch, channels)
            out = torch.mean(out, dim=2)

        out = self.linear_dropout(out)
        out = self.linear(out)
        assert out.size() == (batch_size, 1)

        return out


class Manager(ModelManager):
    def train(
        self,
        data_dir: Path,
        n: Optional[int],
        device: torch.device,
        hp: HyperParameters,
        submission: bool,
    ):
        if not isinstance(hp, SigCnnHp):
            raise ValueError("wrong hyper-parameter class: {hp}")

        wandb.init(project="g2net-" + __name__, entity="wisdom", config=asdict(hp))
        self._train(
            SigCnn(device, hp), device, data_dir, n, [filter_sig_meta], hp, submission
        )
