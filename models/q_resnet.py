from dataclasses import dataclass, asdict
from typing import Type, Dict

import torch
import wandb
from datargs import argsclass
from torch import nn, Tensor

from gw_data import *
from models import HyperParameters, ModelManager
from preprocessor_meta import Preprocessor, qtransform3_meta


def to_odd(i: int) -> int:
    return (i // 2) * 2 + 1


@argsclass(name="q_resnet")
@dataclass
class QResnetHp(HyperParameters):
    batch: int = 64
    epochs: int = 100
    lr: float = 0.001
    dtype: torch.dtype = torch.float32
    convbn1h: int = 5
    convbn1w: int = 5
    convbn2h: int = 5
    convbn2w: int = 5
    convbn3h: int = 3
    convbn3w: int = 3
    convskiph: int = 3
    convskipw: int = 3
    mp: int = 2

    block1out: int = 128
    block2out: int = 256
    block3out: int = 512
    block4out: int = 512

    linear1out = 1

    @property
    def manager_class(self) -> Type[ModelManager]:
        return Manager

    def __post_init__(self):
        self.convbn1h = to_odd(self.convbn1h)
        self.convbn1w = to_odd(self.convbn1w)
        self.convbn2h = to_odd(self.convbn2h)
        self.convbn2w = to_odd(self.convbn2w)
        self.convbn3h = to_odd(self.convbn3h)
        self.convbn3w = to_odd(self.convbn3w)
        self.convskiph = to_odd(self.convskiph)
        self.convskipw = to_odd(self.convskipw)


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

    def forward(self, x: Tensor, use_activation=False):
        out = self.conv(x)
        out = self.bn(out)
        if use_activation is True:
            out = self.activation(out)
        return out


class ResnetBlock(nn.Module):
    def __init__(self, device: torch.device, hp: QResnetHp, in_channels, out_channels):
        super().__init__()
        self.hp = hp
        self.device = device
        self.conv_bn1 = ConvBlock(
            in_channels,
            out_channels,
            (self.hp.convbn1h, self.hp.convbn1w),
            (self.hp.convbn1h // 2, self.hp.convbn1w // 2),
        )
        self.conv_bn2 = ConvBlock(
            out_channels,
            out_channels,
            (self.hp.convbn2h, self.hp.convbn2w),
            (self.hp.convbn2h // 2, self.hp.convbn2w // 2),
        )
        self.conv_bn3 = ConvBlock(
            out_channels,
            out_channels,
            (self.hp.convbn3h, self.hp.convbn3w),
            (self.hp.convbn3h // 2, self.hp.convbn3w // 2),
        )
        self.conv_skip = ConvBlock(
            in_channels,
            out_channels,
            (self.hp.convskiph, self.hp.convskipw),
            (self.hp.convskiph // 2, self.hp.convskipw // 2),
        )
        self.activation = nn.ReLU()
        self.mp = nn.MaxPool2d((self.hp.mp, self.hp.mp))

    def forward(self, x: Tensor):
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

        self.block1 = ResnetBlock(device, hp, N_SIGNALS, self.hp.block1out)
        self.block2 = ResnetBlock(device, hp, self.hp.block1out, self.hp.block2out)
        self.block3 = ResnetBlock(device, hp, self.hp.block2out, self.hp.block3out)
        self.block4 = ResnetBlock(device, hp, self.hp.block3out, self.hp.block4out)

        self.linear1 = nn.Linear(
            in_features=self.hp.block4out,
            out_features=self.hp.linear1out,
        )

    def forward(self, xd: Dict[str, Tensor]) -> Tensor:
        x = xd[qtransform3_meta.name]
        batch_size = x.size()[0]
        assert x.size()[1:] == qtransform3_meta.output_shape
        out = self.block1.forward(x)
        assert out.size()[:2] == (batch_size, self.hp.block1out)
        out = self.block2.forward(out)
        out = self.block3.forward(out)
        out = self.block4.forward(out)
        # Average across h and w, leaving (batch, channels)
        out = torch.mean(out, dim=[2, 3])
        out = self.linear1(out)
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
        if not isinstance(hp, QResnetHp):
            raise ValueError("wrong hyper-parameter class: {hp}")

        wandb.init(project="g2net-" + __name__, entity="wisdom", config=asdict(hp))
        self._train(
            CnnResnet(device, hp),
            device,
            data_dir,
            n,
            [qtransform3_meta],
            hp,
            submission,
        )
