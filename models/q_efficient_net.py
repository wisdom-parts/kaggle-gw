from dataclasses import dataclass, asdict
from enum import Enum, auto
from typing import Type, Dict

import torch
import wandb
import efficientnet_pytorch
from datargs import argsclass
from torch import nn, Tensor

from preprocessor_meta import Preprocessor, qtransform3_meta
from gw_data import *
from models import HyperParameters, ModelManager


@argsclass(name="q_eff")
@dataclass
class QEfficientNetHp(HyperParameters):
    batch: int = 64
    epochs: int = 100
    lr: float = 0.0003
    dtype: torch.dtype = torch.float32

    @property
    def manager_class(self) -> Type[ModelManager]:
        return Manager


class EfficientNet(nn.Module):
    """
    Applies a CNN to the output of preprocess qtransform and produces one logit as output.
    input size: (batch_size, ) + preprocess.qtransform.OUTPUT_SHAPE
    output size: (batch_size, 1)
    """

    def __init__(self, device: torch.device, hp: QEfficientNetHp):
        super().__init__()
        self.hp = hp
        self.device = device
        self.net = efficientnet_pytorch.EfficientNet.from_pretrained("efficientnet-b7")
        n_features = self.net._fc.in_features
        self.net._fc = nn.Linear(in_features=n_features, out_features=1, bias=True)

    def forward(self, xd: Dict[str, Tensor]) -> Tensor:
        x = xd[qtransform3_meta.name]
        assert x.size()[1:] == qtransform3_meta.output_shape
        out = self.net(x)
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
        if not isinstance(hp, QEfficientNetHp):
            raise ValueError("wrong hyper-parameter class: {hp}")

        wandb.init(project="g2net-" + __name__, entity="wisdom", config=asdict(hp))
        self._train(
            EfficientNet(device, hp),
            device,
            data_dir,
            n,
            [qtransform3_meta],
            hp,
            submission,
        )
