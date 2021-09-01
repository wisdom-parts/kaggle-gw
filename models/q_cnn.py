from dataclasses import dataclass, asdict
from typing import Type

import torch
import wandb
from datargs import argsclass
from torch import nn, Tensor

import qtransform_params
from gw_data import *
from models import HyperParameters, ModelManager


@argsclass(name="q_cnn")
@dataclass
class QCnnHp(HyperParameters):
    batch_size: int = 64
    epochs: int = 100
    lr: float = 0.0003
    dtype: torch.dtype = torch.float32

    conv1_h: int = 3  # must be odd
    conv1_w: int = 3  # must be odd
    conv1_out_channels: int = 10
    mp1_h: int = 2
    mp1_w: int = 2

    conv2_h: int = 3  # must be odd
    conv2_w: int = 3  # must be odd
    conv2_out_channels: int = 10
    mp2_h: int = 2
    mp2_w: int = 2

    conv3_h: int = 3  # must be odd
    conv3_w: int = 3  # must be odd
    conv3_out_channels: int = 10
    mp3_h: int = 2
    mp3_w: int = 2

    conv4_h: int = 3  # must be odd
    conv4_w: int = 3  # must be odd
    conv4_out_channels: int = 10
    mp4_h: int = 2
    mp4_w: int = 2

    linear1_out_features = 20

    @property
    def manager_class(self) -> Type[ModelManager]:
        return Manager


class Cnn(nn.Module):
    """
    Applies a CNN to the output of preprocess qtransform and produces two logits as output.
    input size: (batch_size, ) + preprocess.qtransform.OUTPUT_SHAPE
    output size: (batch_size, 2)
    """

    def __init__(self, device: torch.device, hp: QCnnHp):
        super().__init__()
        self.hp = hp
        self.device = device

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=hp.conv1_out_channels,
            kernel_size=(hp.conv1_h, hp.conv1_w),
            stride=(1, 1),
            padding=(hp.conv1_h // 2, hp.conv1_w // 2),
        )
        self.mp1 = nn.MaxPool2d(
            kernel_size=(hp.mp1_h, hp.mp1_w),
        )
        self.mp1_out_h = qtransform_params.FREQ_STEPS // self.hp.mp1_h
        self.mp1_out_w = qtransform_params.TIME_STEPS // self.hp.mp1_w
        self.conv2 = nn.Conv2d(
            in_channels=hp.conv1_out_channels,
            out_channels=hp.conv2_out_channels,
            kernel_size=(hp.conv1_h, hp.conv1_w),
            stride=(1, 1),
            padding=(hp.conv2_h // 2, hp.conv2_w // 2),
        )
        self.mp2 = nn.MaxPool2d(
            kernel_size=(hp.mp2_h, hp.mp2_w),
        )
        self.mp2_out_h = self.mp1_out_h // self.hp.mp2_h
        self.mp2_out_w = self.mp1_out_w // self.hp.mp2_w
        self.conv3 = nn.Conv2d(
            in_channels=hp.conv2_out_channels,
            out_channels=hp.conv3_out_channels,
            kernel_size=(hp.conv3_h, hp.conv3_w),
            stride=(1, 1),
            padding=(hp.conv3_h // 2, hp.conv3_w // 2),
        )
        self.mp3 = nn.MaxPool2d(
            kernel_size=(hp.mp3_h, hp.mp3_w),
        )
        self.mp3_out_h = self.mp2_out_h // self.hp.mp3_h
        self.mp3_out_w = self.mp2_out_w // self.hp.mp3_w

        self.conv4 = nn.Conv2d(
            in_channels=hp.conv3_out_channels,
            out_channels=hp.conv4_out_channels,
            kernel_size=(hp.conv4_h, hp.conv4_w),
            stride=(1, 1),
            padding=(hp.conv4_h // 2, hp.conv4_w // 2),
        )
        self.mp4 = nn.MaxPool2d(
            kernel_size=(hp.mp4_h, hp.mp4_w),
        )
        self.mp4_out_h = self.mp3_out_h // self.hp.mp4_h
        self.mp4_out_w = self.mp3_out_w // self.hp.mp4_w

        self.linear1 = nn.Linear(
            in_features=hp.conv3_out_channels * self.mp4_out_h * self.mp4_out_w,
            out_features=1,
        )
        self.dp = nn.Dropout(p=0.5)

        self.linear2 = nn.Linear(
            in_features=hp.linear1_out_features,
            out_features=1,
        )
        self.activation = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.size()[0] # x is 64, 3, 32, 128
        assert x.size()[1:] == qtransform_params.OUTPUT_SHAPE

        out = self.activation(self.conv1(x))
        assert out.size() == (
            batch_size,
            self.hp.conv1_out_channels,
            qtransform_params.FREQ_STEPS,
            qtransform_params.TIME_STEPS,
        ) # (64, 20, 32, 128)

        out = self.mp1(out) # (64, 20, 16, 64)
        assert out.size() == (
            batch_size,
            self.hp.conv1_out_channels,
            self.mp1_out_h,
            self.mp1_out_w,
        )

        out = self.activation(self.conv2(out)) # (64, 20, 16, 64)
        assert out.size() == (
            batch_size,
            self.hp.conv2_out_channels,
            self.mp1_out_h,
            self.mp1_out_w,
        )

        out = self.mp2(out) # 64, 20, 8, 32
        assert out.size() == (
            batch_size,
            self.hp.conv2_out_channels,
            self.mp2_out_h,
            self.mp2_out_w,
        )

        out = self.activation(self.conv3(out))
        out = self.mp3(out)
        assert out.size() == (
            batch_size,
            self.hp.conv3_out_channels,
            self.mp3_out_h,
            self.mp3_out_w,
        ) # 64, 128, 8, 32

        # out = self.mp3(out)
        out = self.activation(self.conv4(out))
        out = self.mp4(out)
        assert out.size() == (
            batch_size,
            self.hp.conv4_out_channels,
            self.mp4_out_h,
            self.mp4_out_w,
        )

        out = self.linear1(torch.flatten(out, start_dim=1))
        assert out.size() == (batch_size, 1)

        # out = self.linear2(self.dp(out))
        # assert out.size() == (batch_size, 1)

        return out


class Manager(ModelManager):
    def train(self, sources: List[Path], device: torch.device, hp: HyperParameters):
        if len(sources) != 1:
            raise ValueError("must have exactly one source; got {len(sources)}")
        if not isinstance(hp, QCnnHp):
            raise ValueError("wrong hyper-parameter class: {hp}")

        wandb.init(project="g2net-" + __name__, config=asdict(hp))
        self._train(Cnn(device, hp), device, sources[0], hp)
