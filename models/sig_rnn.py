from dataclasses import dataclass, asdict
from enum import Enum, auto
from typing import Type

import torch
import wandb
from datargs import argsclass
from torch import nn, Tensor

from gw_data import *
from models import ModelManager, HyperParameters
from preprocessor_meta import Preprocessor, filter_sig_meta


class RnnType(Enum):
    RNN = auto()
    LSTM = auto()
    GRU = auto()


@argsclass(name="sig_rnn")
@dataclass
class SigRnnHp(HyperParameters):
    n_epochs: int = 100
    lr: float = 0.005
    dtype: torch.dtype = torch.float32

    rnn_type: RnnType = RnnType.RNN
    hidden_dim: int = 7
    n_layers: int = 1
    bidirectional: bool = False

    @property
    def manager_class(self) -> Type[ModelManager]:
        return Manager


class Rnn(nn.Module):
    """
    Applies an RNN to the input and produces two logits as output.

    input size: (batch_size, N_SIGNALS, SIGNAL_LEN)
    output size: (batch_size, 2)
    """

    def __init__(self, device: torch.device, hp: SigRnnHp):
        super().__init__()
        self.hp = hp
        self.device = device
        self.num_directions = 2 if self.hp.bidirectional else 1
        self.rnn_out_channels = self.num_directions * self.hp.hidden_dim

        self.rnn: nn.Module
        if hp.rnn_type == RnnType.RNN:
            self.rnn = nn.RNN(
                N_SIGNALS,
                hp.hidden_dim,
                hp.n_layers,
                batch_first=True,
                bidirectional=self.hp.bidirectional,
            )
        elif hp.rnn_type == RnnType.LSTM:
            self.rnn = nn.LSTM(
                N_SIGNALS,
                hp.hidden_dim,
                hp.n_layers,
                batch_first=True,
                bidirectional=self.hp.bidirectional,
            )
        elif hp.rnn_type == RnnType.GRU:
            self.rnn = nn.GRU(
                N_SIGNALS,
                hp.hidden_dim,
                hp.n_layers,
                batch_first=True,
                bidirectional=self.hp.bidirectional,
            )

        # convolution from the RNN's output at each point in the sequence to logits for target= 0 versus 1
        # noinspection PyTypeChecker
        self.conv = nn.Conv1d(
            in_channels=self.rnn_out_channels, out_channels=2, kernel_size=1
        )

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.size()[0]
        assert x.size() == (batch_size, N_SIGNALS, SIGNAL_LEN)

        out, _ = self.rnn(torch.transpose(x, 1, 2))
        assert out.size() == (batch_size, SIGNAL_LEN, self.rnn_out_channels)

        out = self.conv(torch.transpose(out, 1, 2))
        assert out.size() == (batch_size, 2, SIGNAL_LEN)

        out = torch.mean(out, dim=2)
        assert out.size() == (batch_size, 2)

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
        if not isinstance(hp, SigRnnHp):
            raise ValueError("wrong hyper-parameter class: {hp}")
        wandb.init(project="g2net-" + __name__, config=asdict(hp))
        self._train(
            Rnn(device, hp), device, data_dir, n, [filter_sig_meta], hp, submission
        )
