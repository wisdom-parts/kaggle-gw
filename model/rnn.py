from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable

import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader

from gw_util import *
from model import ModelManager, gw_train_and_test_datasets


class RnnType(Enum):
    RNN = auto()
    LSTM = auto()


@dataclass()
class RnnHyperParameters:
    rnn_type: RnnType = RnnType.RNN
    hidden_dim: int = 31
    n_layers: int = 3
    n_epochs: int = 20
    lr: float = 0.01
    bidirectional: bool = True
    dtype: torch.dtype = torch.float32


class Rnn(nn.Module):
    """
    Applies an RNN to the input and produces two logits as output.

    input size: (batch_size, N_SIGNALS, SIGNAL_LEN)
    output size: (batch_size, 2)
    """

    def __init__(self, hp: RnnHyperParameters, device: torch.device):
        super().__init__()
        self.hp = hp
        self.device = device
        self.num_directions = 2 if self.hp.bidirectional else 1
        self.rnn_out_channels = self.num_directions * self.hp.hidden_dim

        if hp.rnn_type == RnnType.RNN:
            self.rnn = nn.RNN(N_SIGNALS, hp.hidden_dim, hp.n_layers,
                              batch_first=True, bidirectional=self.hp.bidirectional)
        else:
            self.rnn = nn.LSTM(N_SIGNALS, hp.hidden_dim, hp.n_layers,
                               batch_first=True, bidirectional=self.hp.bidirectional)

        # convolution from the RNN's output at each point in the sequence to logits for target= 0 versus 1
        # noinspection PyTypeChecker
        self.conv = nn.Conv1d(in_channels=self.rnn_out_channels,
                              out_channels=2,
                              kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.size()[0]
        assert x.size() == (batch_size, N_SIGNALS, SIGNAL_LEN)

        out, _ = self.rnn(torch.transpose(x, 1, 2), self.initial_hidden(batch_size))
        assert out.size() == (batch_size, SIGNAL_LEN, self.rnn_out_channels)

        out = self.conv(torch.transpose(out, 1, 2))
        assert out.size() == (batch_size, 2, SIGNAL_LEN)

        out = torch.mean(out, dim=2)
        assert out.size() == (batch_size, 2)

        return out

    def initial_hidden(self, batch_size: int) -> Tensor:
        return torch.zeros(self.num_directions * self.hp.n_layers,
                           batch_size,
                           self.hp.hidden_dim,
                           device=self.device,
                           dtype=self.hp.dtype)


class RnnManager(ModelManager):
    def train(self,
              source: Path,
              device: torch.device):

        hp = RnnHyperParameters()

        def transform(x: np.ndarray) -> torch.Tensor:
            return torch.tensor(x, dtype=hp.dtype, device=device)

        def target_transform(y: int) -> torch.Tensor:
            return torch.tensor(y, dtype=torch.long, device=device)

        train_dataset, test_dataset = gw_train_and_test_datasets(source, transform, target_transform)

        model = Rnn(hp, device=device)
        model.to(device, dtype=hp.dtype)

        loss_fn = nn.CrossEntropyLoss()

        train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

        print(hp)

        for epoch in range(hp.n_epochs):
            print(f"---------------- Epoch {epoch + 1} ----------------")
            self.train_epoch(model, loss_fn, train_dataloader, len(train_dataset), hp)
            self.test(model, loss_fn, test_dataloader, len(test_dataset))

        print("Done!")

    @staticmethod
    def train_epoch(model: Rnn, loss_fn: Callable[[Tensor, Tensor], Tensor],
                    dataloader: DataLoader, num_examples: int,
                    hp: RnnHyperParameters):
        optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr)

        for batch_num, (X, y) in enumerate(dataloader):
            # Compute prediction and loss
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_num % 100 == 0:
                loss_val = loss.item()
                i = batch_num * len(X)
                print(f"training loss: {loss_val:>7f}  [{i:>5d}/{num_examples:>5d}]")

    @staticmethod
    def test(model: Rnn, loss_fn: Callable[[Tensor, Tensor], Tensor],
             dataloader: DataLoader, num_examples: int):
        num_batches = len(dataloader)
        test_loss = 0.0
        correct = 0.0

        with torch.no_grad():
            for X, y in dataloader:
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= num_examples
        print(f"----\ntest metrics: Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
