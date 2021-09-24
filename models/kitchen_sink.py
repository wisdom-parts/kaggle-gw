from dataclasses import dataclass, asdict
from typing import Type, Dict

import torch
import wandb
from datargs import argsclass
from torch import nn, Tensor

from gw_data import *
from models import (
    HyperParameters,
    ModelManager,
    q_cnn,
    sig_cnn,
    HpWithRegressionHead,
    RegressionHead,
)
from preprocessor_meta import qtransform_meta, filter_sig_meta


@argsclass(name="kitchen_sink")
@dataclass
class KitchenSinkHp(HpWithRegressionHead):
    batch: int = 250
    epochs: int = 1
    lr: float = 0.001
    dtype: torch.dtype = torch.float32

    linear1drop: float = 0.0
    linear1out: int = 100  # if this value is 1, then omit linear2
    head: RegressionHead = RegressionHead.LINEAR

    @property
    def manager_class(self) -> Type[ModelManager]:
        return Manager

    def q_cnn_hp(self):
        return q_cnn.QCnnHp()

    def sig_cnn_hp(self):
        return sig_cnn.SigCnnHp()


class Model(nn.Module):
    def __init__(self, hp: KitchenSinkHp):
        super().__init__()

        if hp.head != RegressionHead.LINEAR:
            raise NotImplementedError(
                "kitchen_sink only supports RegressionHead.LINEAR"
            )

        self.q_conv = q_cnn.Cnn(hp.q_cnn_hp(), True)
        self.sig_conv = sig_cnn.Cnn(hp.sig_cnn_hp(), True)
        self.linear1 = nn.Linear(
            int(np.prod(self.q_conv.output_shape))
            + int(np.prod(self.sig_conv.output_shape)),
            hp.linear1out,
        )
        self.lin1_dropout = nn.Dropout(p=hp.linear1drop)
        self.lin1_bn = nn.BatchNorm1d(hp.linear1out)
        self.linear_activation = nn.ReLU()
        self.linear2 = nn.Linear(hp.linear1out, 1)

    def forward(self, xd: Dict[str, Tensor]) -> Tensor:
        q_conv_out = self.q_conv(xd[qtransform_meta.name])
        sig_conv_out = self.sig_conv(xd[filter_sig_meta.name])
        out = torch.cat(
            [
                torch.flatten(q_conv_out, start_dim=1),
                torch.flatten(sig_conv_out, start_dim=1),
            ],
            dim=1,
        )
        out = self.linear1(out)
        out = self.lin1_bn(out)
        out = self.linear_activation(out)
        out = self.lin1_dropout(out)
        out = self.linear2(out)
        return out


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

        model = Model(hp)

        self._train(model, device, data_dir, n, [qtransform_meta, filter_sig_meta], hp)
