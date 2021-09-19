from dataclasses import dataclass, asdict
from enum import Enum, auto
from typing import Type, Tuple, Union

import torch
import wandb
from datargs import argsclass
from torch import nn, Tensor

import preprocessor_meta
from gw_data import *
from preprocessor_meta import Preprocessor
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
    batch: int = 512
    epochs: int = 3
    lr: float = 0.0005
    dtype: torch.dtype = torch.float32

    preprocessor: Preprocessor = Preprocessor.QTRANSFORM

    convlayers: int = 4

    conv1h: int = 33
    conv1w: int = 7
    conv1out: int = 10
    mp1h: int = 2
    mp1w: int = 2

    conv2h: int = 17
    conv2w: int = 65
    conv2out: int = 100
    mp2h: int = 2
    mp2w: int = 2

    conv3h: int = 9
    conv3w: int = 16
    conv3out: int = 30
    mp3h: int = 2
    mp3w: int = 2

    conv4h: int = 5
    conv4w: int = 9
    conv4out: int = 100
    mp4h: int = 2
    mp4w: int = 2

    convdrop: float = 0.5

    head: RegressionHead = RegressionHead.MAX

    linear1out: int = 50  # if this value is 1, then omit linear2
    linear1drop: float = 0.4

    @property
    def manager_class(self) -> Type[ModelManager]:
        return Manager

    def __post_init__(self):
        if self.convlayers < 1 or self.convlayers > 4:
            raise ValueError(
                "convlayers must be between 1 and 4; was {self.convlayers}"
            )

        self.conv1h = to_odd(self.conv1h)
        self.conv1w = to_odd(self.conv1w)

        self.conv2h = to_odd(self.conv2h)
        self.conv2w = to_odd(self.conv2w)

        self.conv3h = to_odd(self.conv3h)
        self.conv3w = to_odd(self.conv3w)

        self.conv4h = to_odd(self.conv4h)
        self.conv4w = to_odd(self.conv4w)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, mp_size):
        super().__init__()
        self.mp_size = mp_size
        self.stride = stride

        if kernel_size[0] % 2 == 0 or kernel_size[1] % 2 == 0:
            raise ValueError(f"mp_size must be odd; was {kernel_size}")
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.mp = nn.MaxPool2d(mp_size) if mp_size[0] > 1 or mp_size[1] > 1 else None
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x, use_activation=False):
        out = self.conv(x)
        out = self.bn(out)
        if use_activation is True:
            out = self.activation(out)
        if self.mp:
            out = self.mp(out)
        return out

    def out_size(self, in_size: Tuple[int, int]) -> Tuple[int, int]:
        def s(i: int) -> int:
            return in_size[i] // self.mp_size[i] // self.stride[i]

        return s(0), s(1)


class Cnn(nn.Module):
    """
    Applies a CNN to qtransform data and produces an output shaped like
    (batch, channels, height, width). The size of the output depends on
    hyper-parameters.
    """

    def __init__(self, device: torch.device, hp: QCnnHp, apply_final_activation: bool):
        super().__init__()
        self.device = device
        self.hp = hp
        self.apply_final_activation = apply_final_activation

        self.cb1 = ConvBlock(
            in_channels=N_SIGNALS,
            out_channels=hp.conv1out,
            kernel_size=(hp.conv1h, hp.conv1w),
            stride=(1, 1),
            mp_size=(hp.mp1h, hp.mp1w),
        )

        in_size: Tuple[int, int] = (
            hp.preprocessor.value.output_shape[1],
            hp.preprocessor.value.output_shape[2],
        )
        self.conv_out_size = self.cb1.out_size(in_size)
        if hp.convlayers > 1:
            self.cb2 = ConvBlock(
                in_channels=hp.conv1out,
                out_channels=hp.conv2out,
                kernel_size=(hp.conv2h, hp.conv2w),
                stride=(1, 1),
                mp_size=(hp.mp2h, hp.mp2w),
            )
            self.conv_out_size = self.cb2.out_size(self.conv_out_size)

        if hp.convlayers > 2:
            self.cb3 = ConvBlock(
                in_channels=hp.conv2out,
                out_channels=hp.conv3out,
                kernel_size=(hp.conv3h, hp.conv3w),
                stride=(1, 1),
                mp_size=(hp.mp3h, hp.mp3w),
            )
            self.conv_out_size = self.cb3.out_size(self.conv_out_size)

        if hp.convlayers > 3:
            self.cb4 = ConvBlock(
                in_channels=hp.conv3out,
                out_channels=hp.conv4out,
                kernel_size=(hp.conv4h, hp.conv4w),
                stride=(1, 1),
                mp_size=(hp.mp4h, hp.mp4w),
            )
            self.conv_out_size = self.cb4.out_size(self.conv_out_size)

        self.conv_dropout = nn.Dropout(p=self.hp.convdrop)

        self.conv_out_features = (
            hp.conv1out
            if hp.convlayers == 1
            else (
                hp.conv2out
                if hp.convlayers == 2
                else (hp.conv3out if hp.convlayers == 3 else hp.conv4out)
            )
        )

        self.output_shape = (self.conv_out_features,) + self.conv_out_size

    def forward(self, x: Tensor) -> Tensor:
        out = self.cb1(
            x, use_activation=self.hp.convlayers > 1 or self.apply_final_activation
        )
        if self.hp.convlayers > 1:
            out = self.cb2(
                out,
                use_activation=self.hp.convlayers > 2 or self.apply_final_activation,
            )
        if self.hp.convlayers > 2:
            out = self.cb3(
                out,
                use_activation=self.hp.convlayers > 3 or self.apply_final_activation,
            )
        if self.hp.convlayers > 3:
            out = self.cb4(out, use_activation=self.apply_final_activation)
        return out


class MaxHead(nn.Module):
    """
    Consumes the output of Cnn (channel, h, w) with no final activation
    and returns the maximum across all outputs for each example
    to produce a single logit with no final activation.
    """

    apply_activation_before_input = False

    def __init__(
        self, device: torch.device, hp: QCnnHp, input_shape: Tuple[int, int, int]
    ):
        super().__init__()

    # noinspection PyMethodMayBeStatic
    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.size()[0]
        out = torch.flatten(x, start_dim=1)
        out = torch.amax(out, dim=1, keepdim=True)
        assert out.size() == (batch_size, 1)
        return out


class LinearHead(nn.Module):
    """
    Consumes the output of Cnn (channel, h, w) with a final activation and
    applies one or two linear layers to produce a single logit with no final activation.
    If hp.head == RegressionHead.AVG_LINEAR, then this module first
    takes the average across (h, w).
    """

    apply_activation_before_input = True

    def __init__(
        self, device: torch.device, hp: QCnnHp, input_shape: Tuple[int, int, int]
    ):
        super().__init__()
        self.hp = hp

        linear_input_features = input_shape[0] * (
            1
            if hp.head == RegressionHead.AVG_LINEAR
            else input_shape[1] * input_shape[2]
        )

        self.linear1 = nn.Linear(
            in_features=linear_input_features,
            out_features=hp.linear1out,
        )
        self.lin1_dropout = nn.Dropout(p=self.hp.linear1drop)

        self.activation = nn.ReLU()
        if hp.linear1out > 1:
            self.bn = nn.BatchNorm1d(hp.linear1out)
            self.linear2 = nn.Linear(
                in_features=hp.linear1out,
                out_features=1,
            )

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.size()[0]

        if self.hp.head == RegressionHead.AVG_LINEAR:
            # Average across h and w, leaving (batch, channels)
            out = torch.mean(x, dim=[2, 3])
        else:
            out = torch.flatten(x, start_dim=1)

        out = self.linear1(out)
        if self.hp.linear1out > 1:
            out = self.bn(out)
            out = self.activation(out)
            if self.hp.linear1drop > 0.0:
                out = self.lin1_dropout(out)
            out = self.linear2(out)

        assert out.size() == (batch_size, 1)
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
        prep_test_data: bool,
    ):
        if not isinstance(hp, QCnnHp):
            raise ValueError("wrong hyper-parameter class: {hp}")

        wandb.init(project="g2net-" + __name__, entity="wisdom", config=asdict(hp))

        head_class: Union[Type[MaxHead], Type[LinearHead]] = (
            MaxHead if hp.head == RegressionHead.MAX else LinearHead
        )
        cnn = Cnn(device, hp, head_class.apply_activation_before_input)
        head = head_class(device, hp, cnn.output_shape)
        model = Model(cnn, head)

        self._train(model, device, data_dir, n, [hp.preprocessor.value], hp, prep_test_data)
