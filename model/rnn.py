from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader

from model import ModelManager, GwDataset
from gw_util import *


@dataclass()
class RnnHyperParameters:
    hidden_dim: int = 32
    n_layers: int = 1
    n_epochs: int = 20
    lr: float = 0.01
    bidirectional: bool = True


class Rnn(nn.Module):
    def __init__(self, hp: RnnHyperParameters):
        super().__init__()
        self.hp = hp
        self.num_directions = 2 if self.hp.bidirectional else 1
        self.rnn_out_channels = self.num_directions * self.hp.hidden_dim

        self.rnn = nn.RNN(N_SIGNALS, hp.hidden_dim, hp.n_layers, batch_first=True)

        # convolution from the RNN's output at each point in the sequence to logits for target= 0 versus 1
        # noinspection PyTypeChecker
        self.conv = nn.Conv1d(in_channels=self.rnn_out_channels,
                              out_channels=2,
                              kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.size()[0]
        assert x.size() == (batch_size, SIGNAL_LEN, N_SIGNALS)

        out, _ = self.rnn(x, self.initial_hidden(batch_size),
                          batch_first=True, bidirectional=self.hp.bidirectional)
        assert out.size() == (batch_size, SIGNAL_LEN, self.rnn_out_channels)

        out = self.conv(out)
        assert out.size() == (batch_size, SIGNAL_LEN, 2)

        out = torch.mean(out, dim=1)
        assert out.size() == (batch_size, 2)

        return out

    def initial_hidden(self, batch_size: int) -> Tensor:
        return torch.zeros(self.num_directions, self.hp.n_layers,
                           batch_size,
                           self.hp.hidden_dim)


class RnnManager(ModelManager):
    def train(self, dataset: GwDataset, device: torch.device):
        num_examples = len(dataset)

        hp = RnnHyperParameters()
        model = Rnn(hp)
        model.to(device)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr)

        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

        for batch, (X, y) in enumerate(dataloader):
            # Compute prediction and loss
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{num_examples:>5d}]")
