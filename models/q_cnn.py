from dataclasses import dataclass, asdict
from enum import Enum, auto
from typing import Type

import torch
import wandb
from datargs import argsclass
from torch import nn, Tensor

import qtransform_params
from gw_data import *
from models import HyperParameters, ModelManager


class RegressionHead(Enum):
    LINEAR = auto()
    MAX = auto()
    AVG_LINEAR = auto()


def to_odd(i: int) -> int:
    return (i // 2) * 2 + 1


@argsclass(name="q_cnn")
@dataclass
class QCnnHp(HyperParameters):
    batch: int = 64
    epochs: int = 100
    lr: float = 0.0003
    dtype: torch.dtype = torch.float32

    conv1h: int = 11
    conv1w: int = 11
    conv1out: int = 5
    mp1h: int = 2
    mp1w: int = 2

    conv2h: int = 11
    conv2w: int = 3
    conv2out: int = 80
    mp2h: int = 2
    mp2w: int = 2

    conv3h: int = 7
    conv3w: int = 11
    conv3out: int = 40
    mp3h: int = 2
    mp3w: int = 2

    conv4h: int = 7
    conv4w: int = 7
    conv4out: int = 10
    mp4h: int = 2
    mp4w: int = 2

    convdrop: float = 0.5

    head: RegressionHead = RegressionHead.LINEAR

    linear1out: int = 10  # if this value is 1, then omit linear2
    linear1drop: float = 0.5

    @property
    def manager_class(self) -> Type[ModelManager]:
        return Manager

    def __post_init__(self):
        self.conv1h = to_odd(self.conv1h)
        self.conv1w = to_odd(self.conv1w)

        self.conv2h = to_odd(self.conv2h)
        self.conv2w = to_odd(self.conv2w)

        self.conv3h = to_odd(self.conv3h)
        self.conv3w = to_odd(self.conv3w)

        self.conv4h = to_odd(self.conv4h)
        self.conv4w = to_odd(self.conv4w)


class Cnn(nn.Module):
    """
    Applies a CNN to the output of preprocess qtransform and produces one logit as output.
    input size: (batch_size, ) + preprocess.qtransform.OUTPUT_SHAPE
    output size: (batch_size, 1)
    """

    def __init__(self, device: torch.device, hp: QCnnHp):
        super().__init__()
        self.hp = hp
        self.device = device

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=hp.conv1out,
            kernel_size=(hp.conv1h, hp.conv1w),
            stride=(1, 1),
            padding=(hp.conv1h // 2, hp.conv1w // 2),
        )
        self.mp1 = nn.MaxPool2d(
            kernel_size=(hp.mp1h, hp.mp1w),
        )
        self.mp1_out_h = qtransform_params.FREQ_STEPS // self.hp.mp1h
        self.mp1_out_w = qtransform_params.TIME_STEPS // self.hp.mp1w
        self.bn1 = nn.BatchNorm2d(hp.conv1out)

        self.conv2 = nn.Conv2d(
            in_channels=hp.conv1out,
            out_channels=hp.conv2out,
            kernel_size=(hp.conv2h, hp.conv2w),
            stride=(1, 1),
            padding=(hp.conv2h // 2, hp.conv2w // 2),
        )
        self.mp2 = nn.MaxPool2d(
            kernel_size=(hp.mp2h, hp.mp2w),
        )
        self.mp2_out_h = self.mp1_out_h // self.hp.mp2h
        self.mp2_out_w = self.mp1_out_w // self.hp.mp2w
        self.bn2 = nn.BatchNorm2d(hp.conv2out)

        self.conv3 = nn.Conv2d(
            in_channels=hp.conv2out,
            out_channels=hp.conv3out,
            kernel_size=(hp.conv3h, hp.conv3w),
            stride=(1, 1),
            padding=(hp.conv3h // 2, hp.conv3w // 2),
        )
        self.mp3 = nn.MaxPool2d(
            kernel_size=(hp.mp3h, hp.mp3w),
        )
        self.mp3_out_h = self.mp2_out_h // self.hp.mp3h
        self.mp3_out_w = self.mp2_out_w // self.hp.mp3w
        self.bn3 = nn.BatchNorm2d(hp.conv3out)

        self.conv4 = nn.Conv2d(
            in_channels=hp.conv3out,
            out_channels=hp.conv4out,
            kernel_size=(hp.conv4h, hp.conv4w),
            stride=(1, 1),
            padding=(hp.conv4h // 2, hp.conv4w // 2),
        )
        self.mp4 = nn.MaxPool2d(
            kernel_size=(hp.mp4h, hp.mp4w),
        )
        self.mp4_out_h = self.mp3_out_h // self.hp.mp4h
        self.mp4_out_w = self.mp3_out_w // self.hp.mp4w
        self.bn4 = nn.BatchNorm2d(hp.conv4out)

        self.conv_dropout = nn.Dropout(p=self.hp.convdrop)

        self.flattened_conv_features = (
            hp.conv4out
            if hp.head == RegressionHead.AVG_LINEAR
            else hp.conv4out * self.mp4_out_h * self.mp4_out_w
        )

        self.linear1 = nn.Linear(
            in_features=self.flattened_conv_features,
            out_features=hp.linear1out,
        )
        self.lin1_dropout = nn.Dropout(p=self.hp.linear1drop)

        self.linear2 = nn.Linear(
            in_features=hp.linear1out,
            out_features=1,
        )
        self.conv_activation = nn.ReLU()
        self.linear_activation = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.size()[0]  # x is 64, 3, 32, 128
        assert x.size()[1:] == qtransform_params.OUTPUT_SHAPE

        out = self.bn1(self.conv1(x))
        assert out.size() == (
            batch_size,
            self.hp.conv1out,
            qtransform_params.FREQ_STEPS,
            qtransform_params.TIME_STEPS,
        )  # (64, 20, 32, 128)

        out = self.mp1(self.conv_activation(out))  # (64, 20, 16, 64)
        assert out.size() == (
            batch_size,
            self.hp.conv1out,
            self.mp1_out_h,
            self.mp1_out_w,
        )

        out = self.bn2(self.conv2(out))  # (64, 20, 16, 64)
        assert out.size() == (
            batch_size,
            self.hp.conv2out,
            self.mp1_out_h,
            self.mp1_out_w,
        )

        out = self.mp2(self.conv_activation(out))  # 64, 20, 8, 32
        assert out.size() == (
            batch_size,
            self.hp.conv2out,
            self.mp2_out_h,
            self.mp2_out_w,
        )

        out = self.bn3(self.conv3(out))
        assert out.size() == (
            batch_size,
            self.hp.conv3out,
            self.mp2_out_h,
            self.mp2_out_w,
        )

        out = self.mp3(self.conv_activation(out))
        assert out.size() == (
            batch_size,
            self.hp.conv3out,
            self.mp3_out_h,
            self.mp3_out_w,
        )  # 64, 128, 8, 32

        out = self.bn4(self.conv4(out))
        assert out.size() == (
            batch_size,
            self.hp.conv4out,
            self.mp3_out_h,
            self.mp3_out_w,
        )

        if self.hp.head == RegressionHead.LINEAR:
            out = self.conv_activation(out)

        out = self.mp4(out)
        assert out.size() == (
            batch_size,
            self.hp.conv4out,
            self.mp4_out_h,
            self.mp4_out_w,
        )

        if self.hp.head == RegressionHead.AVG_LINEAR:
            # Average across h and w, leaving (batch, channels)
            out = torch.mean(out, dim=[2, 3])
        else:
            # Keep (c, h, w) dimensions as input to linear or max.
            out = torch.flatten(out, start_dim=1)

        if self.hp.convdrop > 0.0:
            out = self.conv_dropout(out)

        assert out.size() == (batch_size, self.flattened_conv_features)

        if self.hp.head == RegressionHead.MAX:
            out = torch.amax(out, dim=1, keepdim=True)
        else:
            out = self.linear1(self.conv_activation(out))
            assert out.size() == (batch_size, self.hp.linear1out)

            if self.hp.linear1out > 1:
                out = self.linear_activation(out)
                if self.hp.linear1drop > 0.0:
                    out = self.lin1_dropout(out)
                out = self.linear2(out)

        assert out.size() == (batch_size, 1)
        return out


class Manager(ModelManager):
    def train(self, sources: List[Path], device: torch.device, hp: HyperParameters):
        if len(sources) != 1:
            raise ValueError("must have exactly one source; got {len(sources)}")
        if not isinstance(hp, QCnnHp):
            raise ValueError("wrong hyper-parameter class: {hp}")

        wandb.init(project="g2net-" + __name__, entity="wisdom", config=asdict(hp))
        self._train(Cnn(device, hp), device, sources[0], hp)
