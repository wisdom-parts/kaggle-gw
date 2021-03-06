from dataclasses import dataclass, asdict
from typing import Type, Tuple, Union, Dict

import torch
import wandb
from datargs import argsclass
from torch import nn, Tensor

from gw_data import *
from preprocessor_meta import Preprocessor, PreprocessorMeta
from models import (
    HyperParameters,
    ModelManager,
    RegressionHead,
    to_odd,
    MaxHead,
    LinearHead,
    HpWithRegressionHead,
)


@argsclass(name="q_cnn2")
@dataclass
class QCnn2Hp(HpWithRegressionHead):
    batch: int = 400
    epochs: int = 2
    lr: float = 0.015
    dtype: torch.dtype = torch.float32

    linear1drop: float = 0.0
    linear1out: int = 180  # if this value is 1, then omit linear2
    head: RegressionHead = RegressionHead.LINEAR

    preprocessor: Preprocessor = Preprocessor.QTRANSFORM3

    convlayers: int = 4

    tallconv: int = 3

    conv1w: int = 5
    conv1stridew: int = 1
    conv1out: int = 330
    mp1w: int = 1

    conv2w: int = 13
    conv2stridew: int = 1
    conv2out: int = 750
    mp2w: int = 2

    conv3w: int = 9
    conv3stridew: int = 2
    conv3out: int = 240
    mp3w: int = 2

    conv4w: int = 9
    conv4stridew: int = 2
    conv4out: int = 280
    mp4w: int = 2

    @property
    def manager_class(self) -> Type[ModelManager]:
        return Manager

    def __post_init__(self):
        if self.convlayers < 1 or self.convlayers > 4:
            raise ValueError(
                "convlayers must be between 1 and 4; was {self.convlayers}"
            )

        self.conv1w = to_odd(self.conv1w)
        self.conv2w = to_odd(self.conv2w)
        self.conv3w = to_odd(self.conv3w)
        self.conv4w = to_odd(self.conv4w)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int],
        mp_width: int,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.mp_width = mp_width

        if kernel_size[1] % 2 == 0:
            raise ValueError(f"kernel width must be odd; was {kernel_size}")
        padding = (0, kernel_size[1] // 2)

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.mp = nn.MaxPool2d((1, mp_width)) if mp_width > 1 else None
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x: Tensor, use_activation=False):
        out = self.conv(x)
        out = self.bn(out)
        if use_activation is True:
            out = self.activation(out)
        if self.mp:
            out = self.mp(out)
        return out

    def out_size(self, in_size: Tuple[int, int]) -> Tuple[int, int]:
        if self.kernel_size[0] == 1:
            if self.stride[0] != 1:
                raise ValueError(
                    f"given kernel height of 1, vertical stride must be 1; stride={self.stride}"
                )
            s0 = in_size[0]
        elif self.kernel_size[0] == in_size[0]:
            if self.stride[0] != in_size[0]:
                raise ValueError(
                    f"given kernel height of in_size[0], vertical stride must be the same; stride={self.stride}"
                )
            s0 = 1
        else:
            raise ValueError("kernel height must be either 1 or input height")
        return s0, in_size[1] // self.stride[1] // self.mp_width


class Cnn2(nn.Module):
    """
    Applies a 2d CNN "edge on": treats the number of signals as h, time as w, and frequency as channel.
    Produces an output shaped like (batch, channels, height, width). Dimension sizes depend on
    hyper-parameters.
    """

    def __init__(self, hp: QCnn2Hp, apply_final_activation: bool):
        super().__init__()
        self.hp = hp
        self.apply_final_activation = apply_final_activation

        preprocessor_meta: PreprocessorMeta = hp.preprocessor.value
        in_shape = preprocessor_meta.output_shape

        # We don't need the 0th entry here, but it comes along for the ride.
        height = [
            (in_shape[0] if hp.tallconv == i else 1) for i in range(hp.convlayers + 1)
        ]

        self.cb1 = ConvBlock(
            in_channels=in_shape[1],
            out_channels=hp.conv1out,
            kernel_size=(height[1], hp.conv1w),
            stride=(height[1], hp.conv1stridew),
            mp_width=hp.mp1w,
        )

        conv_in_size: Tuple[int, int] = (
            hp.preprocessor.value.output_shape[0],
            hp.preprocessor.value.output_shape[2],
        )
        self.conv_out_size = self.cb1.out_size(conv_in_size)
        if hp.convlayers > 1:
            self.cb2 = ConvBlock(
                in_channels=hp.conv1out,
                out_channels=hp.conv2out,
                kernel_size=(height[2], hp.conv2w),
                stride=(height[2], hp.conv2stridew),
                mp_width=hp.mp2w,
            )
            self.conv_out_size = self.cb2.out_size(self.conv_out_size)

        if hp.convlayers > 2:
            self.cb3 = ConvBlock(
                in_channels=hp.conv2out,
                out_channels=hp.conv3out,
                kernel_size=(height[3], hp.conv3w),
                stride=(height[3], hp.conv3stridew),
                mp_width=hp.mp3w,
            )
            self.conv_out_size = self.cb3.out_size(self.conv_out_size)

        if hp.convlayers > 3:
            self.cb4 = ConvBlock(
                in_channels=hp.conv3out,
                out_channels=hp.conv4out,
                kernel_size=(height[4], hp.conv4w),
                stride=(height[4], hp.conv4stridew),
                mp_width=hp.mp4w,
            )
            self.conv_out_size = self.cb4.out_size(self.conv_out_size)

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
        # Use frequency as channel and signals as height
        out = x.transpose(1, 2)
        out = self.cb1(
            out, use_activation=self.hp.convlayers > 1 or self.apply_final_activation
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


class Model(nn.Module):
    def __init__(self, cnn: nn.Module, head: nn.Module, hp: QCnn2Hp):
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
        submission: bool,
    ):
        if not isinstance(hp, QCnn2Hp):
            raise ValueError("wrong hyper-parameter class: {hp}")

        wandb.init(project="g2net-" + __name__, entity="wisdom", config=asdict(hp))

        head_class: Union[Type[MaxHead], Type[LinearHead]] = (
            MaxHead if hp.head == RegressionHead.MAX else LinearHead
        )
        cnn = Cnn2(hp, head_class.apply_activation_before_input)
        head = head_class(hp, cnn.output_shape)
        model = Model(cnn, head, hp)

        self._train(model, device, data_dir, n, [hp.preprocessor.value], hp, submission)
