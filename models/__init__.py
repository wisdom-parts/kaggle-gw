import random
from abc import abstractmethod, ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable, List, Dict, Tuple, Type

import numpy as np
import torch
import wandb
from torch import Tensor, nn
from torch.utils.data import Dataset, random_split, DataLoader, Subset

from gw_data import training_labels_file, train_file, validate_source_dir


class GwDataset(Dataset[Tuple[Tensor, Tensor]]):
    """
    Represents the training examples of a g2net data directory as Tensor's
    of size (N_SIGNALS, SIGNAL_LEN).
    """

    def __init__(
        self,
        source: Path,
        transform: Callable[[np.ndarray], Tensor],
        target_transform: Callable[[int], Tensor],
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

        x = self.transform(np.load(fpath))
        y = self.target_transform(self.id_to_label[_id])

        return x, y


@dataclass
class MyDatasets:
    gw: GwDataset
    train: Subset[Tuple[Tensor, Tensor]]
    test: Subset[Tuple[Tensor, Tensor]]


def gw_train_and_test_datasets(
    source: Path, dtype: torch.dtype, device: torch.device
) -> MyDatasets:
    def transform(x: np.ndarray) -> torch.Tensor:
        return torch.tensor(x, dtype=dtype, device=device)

    def target_transform(y: int) -> torch.Tensor:
        return torch.tensor((y,), dtype=dtype, device=device)

    gw = GwDataset(source, transform=transform, target_transform=target_transform)
    num_examples = len(gw)
    num_train_examples = int(num_examples * 0.8)
    num_test_examples = num_examples - num_train_examples
    train, test = random_split(gw, [num_train_examples, num_test_examples])
    return MyDatasets(gw, train, test)


SAMPLES_TO_CHECK = 300
MAX_SAMPLES_PER_KEY = 6


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

    # noinspection PyCallingNonCallable
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
        fp = 0.0
        fn = 0.0
        tp = 0.0
        tn = 0.0

        with torch.no_grad():
            for X, y in dataloader:
                pred = model(X)
                loss = loss_fn(pred, y)

                test_loss += loss.item()

                correct += torch.sum(torch.eq(pred > 0.0, y > 0.0)).item()

                tp += torch.sum(torch.bitwise_and(pred > 0.0, y == 1)).item()
                fp += torch.sum(torch.bitwise_and(pred > 0.0, y == 0)).item()

                tn += torch.sum(torch.bitwise_and(pred < 0.0, y == 0)).item()
                fn += torch.sum(torch.bitwise_and(pred < 0.0, y == 1)).item()

        test_loss /= num_batches
        test_accuracy = 100.0 * correct / num_examples
        print(
            f"----\ntest metrics: Accuracy: {test_accuracy:>0.1f}%, Avg loss: {test_loss:>8f} \n"
        )
        wandb.log(
            {
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "TP": tp,
                "FP": fp,
                "TN": tn,
                "FN": fn,
                "num_examples": num_examples,
            }
        )

    def _train(
        self,
        model: nn.Module,
        device: torch.device,
        source: Path,
        hp: "HyperParameters",
    ):
        data = gw_train_and_test_datasets(source, hp.dtype, device)
        model.to(device, dtype=hp.dtype)
        loss_fn = nn.BCEWithLogitsLoss()
        wandb.watch(model, criterion=loss_fn, log="all", log_freq=100)
        train_dataloader = DataLoader(
            data.train, batch_size=hp.batch_size, shuffle=True
        )
        test_dataloader = DataLoader(data.test, batch_size=hp.batch_size, shuffle=True)
        print(hp)
        optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr)
        for epoch in range(hp.epochs):
            print(f"---------------- Epoch {epoch + 1} ----------------")
            self._train_epoch(
                model, loss_fn, train_dataloader, len(data.train), optimizer
            )
            self._test(model, loss_fn, test_dataloader, len(data.test))

        confusion_sample_indices = (
            random.sample(data.test.indices, SAMPLES_TO_CHECK)
            if len(data.test) > SAMPLES_TO_CHECK
            else data.test.indices
        )
        confusion_sample: Dict[str, List[str]] = {}

        def add_to_sample(key: str, _id: str):
            if key not in confusion_sample:
                confusion_sample[key] = [_id]
            elif len(confusion_sample[key]) < MAX_SAMPLES_PER_KEY:
                confusion_sample[key].append(_id)

        for i in confusion_sample_indices:
            _id = data.gw.ids[i]
            x, y = data.gw[i]
            # add batch dimension
            x = torch.unsqueeze(x, 0)
            y = torch.unsqueeze(y, 0)
            pred = model(x)

            yv = y.item()
            pv = pred.item()
            if yv:
                if pv > 0.0:
                    add_to_sample("tp", _id)
                else:
                    add_to_sample("fn", _id)
            else:
                if pv > 0.0:
                    add_to_sample("fp", _id)
                else:
                    add_to_sample("tn", _id)

        print("Confusion matrix sample:")
        print(repr(confusion_sample))

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
