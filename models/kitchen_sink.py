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


@argsclass(name="kitchen_sink")
@dataclass
class KitchenSinkHp(HyperParameters):
    batch: int = 512
    epochs: int = 3
    lr: float = 0.0005
    dtype: torch.dtype = torch.float32

    @property
    def manager_class(self) -> Type[ModelManager]:
        return Manager


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x


class Manager(ModelManager):
    def train(
        self,
        data_dir: Path,
        n: Optional[int],
        device: torch.device,
        hp: HyperParameters,
    ):
        if not isinstance(hp, KitchenSinkHp):
            raise ValueError("wrong hyper-parameter class: {hp}")

        wandb.init(project="g2net-" + __name__, entity="wisdom", config=asdict(hp))

        model = Model()

        # self._train(model, device, data_dir, n, [hp.preprocessor.value], hp)
