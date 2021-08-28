from dataclasses import dataclass, asdict

import torch
import wandb
from torch import nn, Tensor

import qtransform_params
from gw_data import *
from models import HyperParameters, ModelManager


@dataclass()
class CnnHyperParameters(HyperParameters):
    batch_size: int = 64
    n_epochs: int = 100
    lr: float = 0.0003
    dtype: torch.dtype = torch.float32

    conv1a_out_channels: int = 3
    conv1b_out_channels: int = 5
    mp1_h: int = 2
    mp1_w: int = 2

    conv2_h: int = 5  # must be odd
    conv2_w: int = 5  # must be odd
    conv2_out_channels: int = 5
    mp2_h: int = 3
    mp2_w: int = 4


class Cnn(nn.Module):
    """
    Applies a CNN to the output of preprocess qtransform and produces two logits as output.
    input size: (batch_size, ) + preprocess.qtransform.OUTPUT_SHAPE
    output size: (batch_size, 2)
    """

    def __init__(self, device: torch.device, hp: CnnHyperParameters):
        super().__init__()
        self.hp = hp
        self.device = device

        self.conv1a = nn.Conv2d(
            in_channels=3,
            out_channels=hp.conv1a_out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.conv1b = nn.Conv2d(
            in_channels=hp.conv1a_out_channels,
            out_channels=hp.conv1b_out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.mp1 = nn.MaxPool2d(
            kernel_size=(hp.mp1_h, hp.mp1_w),
        )
        self.conv2 = nn.Conv2d(
            in_channels=hp.conv1b_out_channels,
            out_channels=hp.conv2_out_channels,
            kernel_size=(hp.conv2_h, hp.conv2_w),
            stride=(1, 1),
            padding=(hp.conv2_h // 2, hp.conv2_w // 2),
        )
        self.mp2 = nn.MaxPool2d(
            kernel_size=(hp.mp2_h, hp.mp2_w),
        )

        # Do the size math here, to find out the number of input features for the linear layer.
        self.mp1_out_h = qtransform_params.FREQ_STEPS // self.hp.mp1_h
        self.mp1_out_w = qtransform_params.TIME_STEPS // self.hp.mp1_w

        self.mp2_out_h = self.mp1_out_h // self.hp.mp2_h
        self.mp2_out_w = self.mp1_out_w // self.hp.mp2_w

        self.linear = nn.Linear(
            in_features=hp.conv2_out_channels * self.mp2_out_h * self.mp2_out_w,
            out_features=2,
        )
        self.activation = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.size()[0]
        assert x.size()[1:] == qtransform_params.OUTPUT_SHAPE

        out = self.activation(self.conv1a(x))
        assert out.size() == (
            batch_size,
            self.hp.conv1a_out_channels,
            qtransform_params.FREQ_STEPS,
            qtransform_params.TIME_STEPS,
        )

        out = self.activation(self.conv1b(out))
        assert out.size() == (
            batch_size,
            self.hp.conv1b_out_channels,
            qtransform_params.FREQ_STEPS,
            qtransform_params.TIME_STEPS,
        )

        out = self.mp1(out)
        assert out.size() == (
            batch_size,
            self.hp.conv1b_out_channels,
            self.mp1_out_h,
            self.mp1_out_w,
        )

        out = self.activation(self.conv2(out))
        assert out.size() == (
            batch_size,
            self.hp.conv2_out_channels,
            self.mp1_out_h,
            self.mp1_out_w,
        )

        out = self.mp2(out)
        assert out.size() == (
            batch_size,
            self.hp.conv2_out_channels,
            self.mp2_out_h,
            self.mp2_out_w,
        )

        out = self.activation(self.linear(torch.flatten(out, start_dim=1)))
        assert out.size() == (batch_size, 2)

        return out


class Manager(ModelManager):
    def train(self, source: Path, device: torch.device):
        hp = CnnHyperParameters()
        wandb.init(project="g2net-q_cnn_conventional", config=asdict(hp))
        self._train(Cnn(device, hp), device, source, hp)
