import pdb
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

@argsclass(name="q_resnet")
@dataclass
class QResnetHp(HyperParameters):
    batch_size: int = 64
    epochs: int = 100
    lr: float = 0.0003
    dtype: torch.dtype = torch.float32

    @property
    def manager_class(self) -> Type[ModelManager]:
        return Manager

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding_size):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=(1, 1),
            padding=padding_size,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x, use_activation=False):
        out = self.conv(x)
        out = self.bn(out)
        if use_activation is True:
            out = self.activation(out)
        return out


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_bn1 = ConvBlock(in_channels, out_channels, (5, 5), (2, 2))
        self.conv_bn2 = ConvBlock(out_channels, out_channels, (5, 5), (2, 2))
        self.conv_bn3 = ConvBlock(out_channels, out_channels, (3, 3), (1, 1))
        self.conv_skip = ConvBlock(in_channels, out_channels, (3, 3), (1, 1))
        self.activation = nn.ReLU()
        self.mp = nn.MaxPool2d((2,2))

    def forward(self, x):
        out = self.conv_bn1.forward(x, True)
        out = self.conv_bn2.forward(out, True)
        out = self.conv_bn3.forward(out, False)
        x_skip = self.conv_skip.forward(x, False)
        out = x_skip + out
        return self.mp(self.activation(out))


class CnnResnet(nn.Module):
    """
    Applies a CNN to the output of preprocess qtransform and produces two logits as output.
    input size: (batch_size, ) + preprocess.qtransform.OUTPUT_SHAPE
    output size: (batch_size, 2)
    """

    def __init__(self, device: torch.device, hp: QResnetHp):
        super().__init__()
        self.hp = hp
        self.device = device

        self.block1 = ResnetBlock(3, 256) # out_channels=32 # 32,128
        self.block2 = ResnetBlock(256, 256) # out_channels=64 # 16, 64
        self.block3 = ResnetBlock(256, 256) # 8, 32
        self.block4 = ResnetBlock(256, 256) # 4, 16

        self.avg_pool = nn.AvgPool2d((2, 8))
        self.linear1 = nn.Linear(
            in_features=256,
            out_features=1,
        )


    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.size()[0]  # x is 64, 3, 32, 128
        assert x.size()[1:] == qtransform_params.OUTPUT_SHAPE
        out = self.block1.forward(x) # out here is 64, 32, 128
        out = self.block2.forward(out) # out here is 128, 8, 32
        out = self.block3.forward(out) # 256, 4, 16
        out = self.block4.forward(out) # 512, 2, 8
        out = self.avg_pool(out)
        out = torch.flatten(out, start_dim=1)
        out = self.linear1(out)
        return out


class Manager(ModelManager):
    def train(self, sources: List[Path], device: torch.device, hp: HyperParameters):
        if len(sources) != 1:
            raise ValueError("must have exactly one source; got {len(sources)}")
        if not isinstance(hp, QResnetHp):
            raise ValueError("wrong hyper-parameter class: {hp}")

        wandb.init(project="g2net-" + __name__, entity="wisdom", config=asdict(hp))
        self._train(CnnResnet(device, hp), device, sources[0], hp)
