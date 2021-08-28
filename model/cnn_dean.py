from dataclasses import dataclass, asdict
from typing import Callable

import torch
import wandb
from torch import nn, Tensor
from torch.utils.data import DataLoader

import preprocess.qtransform as q
from gw_util import *
from model import HyperParameters, ModelManager, gw_train_and_test_datasets


@dataclass()
class CnnHyperParameters(HyperParameters):
    batch_size: int = 64
    n_epochs: int = 100
    lr: float = 0.0003
    dtype: torch.dtype = torch.float32

    conv1a_out_channels: int = 3
    conv1b_out_channels: int = 5
    mp1_h: int = 2
    mp1_w: int = 2

    conv2_h: int = 5  # must be odd
    conv2_w: int = 5  # must be odd
    conv2_out_channels: int = 5
    mp2_h: int = 3
    mp2_w: int = 4


class Cnn(nn.Module):
    """
    Applies a CNN to the output of preprocess qtransform and produces two logits as output.
    input size: (batch_size, ) + preprocess.qtransform.OUTPUT_SHAPE
    output size: (batch_size, 2)
    """

    def __init__(self, hp: CnnHyperParameters, device: torch.device):
        super().__init__()
        self.hp = hp
        self.device = device

        self.conv1a = nn.Conv2d(
            in_channels=3,
            out_channels=hp.conv1a_out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.conv1b = nn.Conv2d(
            in_channels=hp.conv1a_out_channels,
            out_channels=hp.conv1b_out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.mp1 = nn.MaxPool2d(
            kernel_size=(hp.mp1_h, hp.mp1_w),
        )
        self.conv2 = nn.Conv2d(
            in_channels=hp.conv1b_out_channels,
            out_channels=hp.conv2_out_channels,
            kernel_size=(hp.conv2_h, hp.conv2_w),
            stride=(1, 1),
            padding=(hp.conv2_h // 2, hp.conv2_w // 2)
        )
        self.mp2 = nn.MaxPool2d(
            kernel_size=(hp.mp2_h, hp.mp2_w),
        )

        # Do the size math here, to find out the number of input features for the linear layer.
        self.mp1_out_h = q.FREQ_STEPS // self.hp.mp1_h
        self.mp1_out_w = q.TIME_STEPS // self.hp.mp1_w

        self.mp2_out_h = self.mp1_out_h // self.hp.mp2_h
        self.mp2_out_w = self.mp1_out_w // self.hp.mp2_w

        self.linear = nn.Linear(
            in_features=hp.conv2_out_channels * self.mp2_out_h * self.mp2_out_w,
            out_features=2,
        )
        self.activation = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.size()[0]
        assert x.size()[1:] == q.OUTPUT_SHAPE

        out = self.activation(self.conv1a(x))
        assert out.size() == (
            batch_size,
            self.hp.conv1a_out_channels,
            q.FREQ_STEPS,
            q.TIME_STEPS,
        )

        out = self.activation(self.conv1b(out))
        assert out.size() == (
            batch_size,
            self.hp.conv1b_out_channels,
            q.FREQ_STEPS,
            q.TIME_STEPS,
        )

        out = self.mp1(out)
        assert out.size() == (
            batch_size,
            self.hp.conv1b_out_channels,
            self.mp1_out_h,
            self.mp1_out_w,
        )

        out = self.activation(self.conv2(out))
        assert out.size() == (
            batch_size,
            self.hp.conv2_out_channels,
            self.mp1_out_h,
            self.mp1_out_w,
        )

        out = self.mp2(out)
        assert out.size() == (
            batch_size,
            self.hp.conv2_out_channels,
            self.mp2_out_h,
            self.mp2_out_w,
        )

        out = self.activation(self.linear(torch.flatten(out, start_dim=1)))
        assert out.size() == (batch_size, 2)

        return out


class Manager(ModelManager):
    def train(self, source: Path, device: torch.device):

        hp = CnnHyperParameters()
        wandb.init(project="g2net-cnn_dean", config=asdict(hp))

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

        wandb.watch(model, criterion=loss_fn, log="all", log_freq=100)

        train_dataloader = DataLoader(
            train_dataset, batch_size=hp.batch_size, shuffle=True
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=hp.batch_size, shuffle=True
        )

        print(hp)

        optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr)

        for epoch in range(hp.n_epochs):
            print(f"---------------- Epoch {epoch + 1} ----------------")
            self.train_epoch(model, loss_fn, train_dataloader, len(train_dataset), hp, optimizer)
            self.test(model, loss_fn, test_dataloader, len(test_dataset))

        print("Done!")

    @staticmethod
    def train_epoch(
        model: Cnn,
        loss_fn: Callable[[Tensor, Tensor], Tensor],
        dataloader: DataLoader,
        num_examples: int,
        hp: CnnHyperParameters,
        optimizer: torch.optim.Optimizer,
    ):
        for batch_num, (X, y) in enumerate(dataloader):

            optimizer.zero_grad()

            pred = model(X)
            loss = loss_fn(pred, y)

            loss.backward()
            optimizer.step()

            wandb.log({"train_pred": pred.detach().cpu().numpy(), "train_loss": loss.item()})
            if batch_num % 100 == 0:
                i = batch_num * len(X)
                print(f"training loss: {loss.item():>7f}  [{i:>5d}/{num_examples:>5d}]")

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
        test_accuracy = 100.0 * correct / num_examples
        print(
            f"----\ntest metrics: Accuracy: {test_accuracy:>0.1f}%, Avg loss: {test_loss:>8f} \n"
        )
        wandb.log({"test_loss": test_loss, "test_accuracy": test_accuracy})
