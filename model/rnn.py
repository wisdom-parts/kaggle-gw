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


class Rnn(nn.Module):
    def __init__(self, hp: RnnHyperParameters):
        super().__init__()
        self.hp = hp

        self.rnn = nn.RNN(N_SIGNALS, hp.hidden_dim, hp.n_layers, batch_first=True)
        self.fc = nn.Linear(hp.hidden_dim, 2)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size = x.size()[0]

        out, hidden = self.rnn(x, self.initial_hidden(batch_size))
        out = out.contiguous().view(-1, self.hp.hidden_dim)
        out = self.fc(out)

        return out, hidden

    def initial_hidden(self, batch_size: int) -> Tensor:
        return torch.zeros(self.hp.n_layers, batch_size, self.hp.hidden_dim)


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
