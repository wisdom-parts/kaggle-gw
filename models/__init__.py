from abc import abstractmethod, ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable, List, Dict, Tuple, Type

import numpy as np
import torch
import wandb
from torch import Tensor, nn
from torch.utils.data import Dataset, random_split, DataLoader

from gw_data import training_labels_file, train_file, validate_source_dir


class GwDataset(Dataset):
    """
    Represents the training examples of a g2net data directory as float32 Tensor's
    of size (N_SIGNALS, SIGNAL_LEN).
    """

    def __init__(
        self,
        source: Path,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        self.source = source
        self.transform = transform
        self.target_transform = target_transform
        self.ids: List[str] = []
        self.id_to_label: Dict[str, int] = {}
        with open(training_labels_file(source)) as id_label_file:
            for id_label in id_label_file:
                _id, label = id_label.split(",")
                if _id != "id":
                    self.ids.append(_id)
                    self.id_to_label[_id] = int(label)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        _id = self.ids[idx]
        fpath = str(train_file(self.source, _id))

        x = np.load(fpath)
        if self.transform:
            x = self.transform(x)

        y = self.id_to_label[_id]
        if self.target_transform:
            y = self.target_transform(y)

        return x, y


def gw_train_and_test_datasets(
    source: Path, transform: Optional[Callable], target_transform: Optional[Callable]
):
    dataset = GwDataset(source, transform=transform, target_transform=target_transform)
    num_examples = len(dataset)
    num_train_examples = int(num_examples * 0.8)
    num_test_examples = num_examples - num_train_examples
    return random_split(dataset, [num_train_examples, num_test_examples])


class ModelManager(ABC):
    @abstractmethod
    def train(self, sources: List[Path], device: torch.device, hp: "HyperParameters"):
        pass

    def _train_epoch(
        self,
        model: nn.Module,
        loss_fn: Callable[[Tensor, Tensor], Tensor],
        dataloader: DataLoader,
        num_examples: int,
        optimizer: torch.optim.Optimizer,
    ):
        for batch_num, (X, y) in enumerate(dataloader):

            optimizer.zero_grad()

            pred = model(X)
            loss = loss_fn(pred, y)

            loss.backward()
            optimizer.step()

            wandb.log(
                {"train_pred": pred.detach().cpu().numpy(), "train_loss": loss.item()}
            )
            if batch_num % 100 == 0:
                i = batch_num * len(X)
                print(f"training loss: {loss.item():>7f}  [{i:>5d}/{num_examples:>5d}]")

    def _test(
        self,
        model: nn.Module,
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
                correct_in_batch = torch.count_nonzero(torch.eq(pred > 0.0, y > 0.0))
                correct += correct_in_batch

        test_loss /= num_batches
        test_accuracy = 100.0 * correct / num_examples
        print(
            f"----\ntest metrics: Accuracy: {test_accuracy:>0.1f}%, Avg loss: {test_loss:>8f} \n"
        )
        wandb.log({"test_loss": test_loss, "test_accuracy": test_accuracy})

    def _train(
        self,
        model: nn.Module,
        device: torch.device,
        source: Path,
        hp: "HyperParameters",
    ):
        def transform(x: np.ndarray) -> torch.Tensor:
            return torch.tensor(x, dtype=hp.dtype, device=device)

        def target_transform(y: int) -> torch.Tensor:
            return torch.tensor((y,), dtype=hp.dtype, device=device)

        train_dataset, test_dataset = gw_train_and_test_datasets(
            source, transform, target_transform
        )
        model.to(device, dtype=hp.dtype)
        loss_fn = nn.BCEWithLogitsLoss()
        wandb.watch(model, criterion=loss_fn, log="all", log_freq=100)
        train_dataloader = DataLoader(
            train_dataset, batch_size=hp.batch_size, shuffle=True
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=hp.batch_size, shuffle=True
        )
        print(hp)
        optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr)
        for epoch in range(hp.epochs):
            print(f"---------------- Epoch {epoch + 1} ----------------")
            self._train_epoch(
                model, loss_fn, train_dataloader, len(train_dataset), optimizer
            )
            self._test(model, loss_fn, test_dataloader, len(test_dataset))
        print("Done!")


@dataclass()
class HyperParameters:
    batch_size: int = 64
    epochs: int = 100
    lr: float = 0.0003
    dtype: torch.dtype = torch.float32

    @property
    def manager_class(self) -> Type[ModelManager]:
        return ModelManager


def train_model(manager: ModelManager, sources: List[Path], hp: HyperParameters):
    for source in sources:
        validate_source_dir(source)

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device {device_name}")
    device = torch.device(device_name)

    manager.train(sources, device, hp)
