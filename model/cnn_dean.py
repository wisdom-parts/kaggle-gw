from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader

import preprocess.qtransform as q
from gw_util import *
from model import ModelManager, gw_train_and_test_datasets


@dataclass()
class HyperParameters:
    batch_size = 65
    n_epochs = 100
    lr = 0.005
    dtype = torch.float32
    conv1_out_channels = 2
    conv2_width = 8
    conv2_h_stride = 4
    conv2_out_channels = 2


class Cnn(nn.Module):
    """
    Applies a CNN to the output of preprocess qtransform and produces two logits as output.
    input size: (batch_size, ) + preprocess.qtransform.OUTPUT_SHAPE
    output size: (batch_size, 2)
    """

    def __init__(self, hp: HyperParameters, device: torch.device):
        super().__init__()
        self.hp = hp
        self.device = device

        # An opportunity for the model to compare and contrast the three signals
        # and find local patterns.
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=hp.conv1_out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        # An opportunity to look for signal in vertical slices.
        # We imagine that each frequency has a different meaning,
        # hence we should work with vertical slices instead of patches.
        self.conv2 = nn.Conv2d(
            in_channels=hp.conv1_out_channels,
            out_channels=hp.conv2_out_channels,
            kernel_size=(q.FREQ_STEPS, hp.conv2_width),
            stride=(q.FREQ_STEPS, hp.conv2_h_stride),
        )
        self.conv2_out_w = 1 + ((q.TIME_STEPS - hp.conv2_width) // hp.conv2_h_stride)
        # We imagine that in this data (unlike real life) specific times
        # in the 2-second interval have meaning, so we use a dense layer
        # across time.
        self.linear = nn.Linear(
            in_features=hp.conv2_out_channels * self.conv2_out_w,
            out_features=2
        )
        self.activation = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.size()[0]
        assert x.size()[1:] == q.OUTPUT_SHAPE

        out = self.activation(self.conv1(x))
        assert out.size() == (batch_size, self.hp.conv1_out_channels, q.FREQ_STEPS, q.TIME_STEPS)

        out = self.activation(self.conv2(out))
        assert out.size() == (batch_size, self.hp.conv2_out_channels, 1, self.conv2_out_w)

        out = self.activation(self.linear(torch.flatten(out, start_dim=1)))
        assert out.size() == (batch_size, 2)

        return out


class Manager(ModelManager):
    def train(self, source: Path, device: torch.device):

        hp = HyperParameters()

        def transform(x: np.ndarray) -> torch.Tensor:
            return torch.tensor(x, dtype=hp.dtype, device=device)

        def target_transform(y: int) -> torch.Tensor:
            return torch.tensor(y, dtype=torch.long, device=device)

        train_dataset, test_dataset = gw_train_and_test_datasets(
            source, transform, target_transform
        )

        model = Cnn(hp, device=device)
        model.to(device, dtype=hp.dtype)

        loss_fn = nn.CrossEntropyLoss()

        train_dataloader = DataLoader(train_dataset, batch_size=hp.batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=hp.batch_size, shuffle=True)

        print(hp)

        for epoch in range(hp.n_epochs):
            print(f"---------------- Epoch {epoch + 1} ----------------")
            self.train_epoch(model, loss_fn, train_dataloader, len(train_dataset), hp)
            self.test(model, loss_fn, test_dataloader, len(test_dataset))

        print("Done!")

    @staticmethod
    def train_epoch(
        model: Cnn,
        loss_fn: Callable[[Tensor, Tensor], Tensor],
        dataloader: DataLoader,
        num_examples: int,
        hp: HyperParameters,
    ):
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
    def test(
        model: Cnn,
        loss_fn: Callable[[Tensor, Tensor], Tensor],
        dataloader: DataLoader,
        num_examples: int,
    ):
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
        print(
            f"----\ntest metrics: Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
        )
