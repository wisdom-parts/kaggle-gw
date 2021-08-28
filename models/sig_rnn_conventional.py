from dataclasses import dataclass, asdict
from enum import Enum, auto

import torch
import wandb
from torch import nn, Tensor

from gw_data import *
from models import ModelManager, HyperParameters


class RnnType(Enum):
    RNN = auto()
    LSTM = auto()
    GRU = auto()


@dataclass()
class RnnHyperParameters(HyperParameters):
    n_epochs: int = 100
    lr: float = 0.005
    dtype: torch.dtype = torch.float32

    rnn_type: RnnType = RnnType.RNN
    hidden_dim: int = 7
    n_layers: int = 1
    bidirectional: bool = False


class Rnn(nn.Module):
    """
    Applies an RNN to the input and produces two logits as output.

    input size: (batch_size, N_SIGNALS, SIGNAL_LEN)
    output size: (batch_size, 2)
    """

    def __init__(self, device: torch.device, hp: RnnHyperParameters):
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
    def train(self, source: Path, device: torch.device):
        hp = RnnHyperParameters()
        wandb.init(project="g2net-sig_rnn_conventional", config=asdict(hp))
        self._train(Rnn(device, hp), device, source, hp)
