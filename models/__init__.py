import csv
import datetime
import pickle
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
from sklearn.metrics import roc_auc_score

from gw_data import training_labels_file, train_file, validate_source_dir, sample_submission_file, test_file
from preprocessor_meta import PreprocessorMeta

class GwSubmissionDataset(Dataset[Tuple[Tensor]]):
    """
    Represents the test data of the g2net data directory as Tensors.
    """
    def __init__(
        self,
        data_dir: Path,
        preprocessors: List[PreprocessorMeta],
        transform: Callable[[np.ndarray], Tensor],
    ):
        if len(preprocessors) > 1:
            raise ValueError("multiple data names not yet supported")
        self.data_dir = data_dir
        self.transform = transform
        self.ids: List[str] = []
        preprocessor_name = preprocessors[0].name
        with open(sample_submission_file(data_dir)) as test_id_label_file:
            for id_label in test_id_label_file:
                _id, _ = id_label.split(",")
                if _id != "id":
                    self.ids.append(_id)
        self.data_name = (
            preprocessor_name
            if test_file(data_dir, self.ids[0], preprocessor_name).exists()
            else None
        )

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int) -> Tensor:
        _id = self.ids[idx]
        fpath = str(test_file(self.data_dir, _id, self.data_name))

        x = self.transform(np.load(fpath))
        return x

class GwDataset(Dataset[Tuple[Tensor, Tensor]]):
    """
    Represents the training examples of a g2net data directory as Tensors.
    """

    def __init__(
        self,
        data_dir: Path,
        n: Optional[int],
        preprocessors: List[PreprocessorMeta],
        transform: Callable[[np.ndarray], Tensor],
        target_transform: Callable[[int], Tensor],
    ):
        if len(preprocessors) > 1:
            raise ValueError("multiple data names not yet supported")
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        self.ids: List[str] = []
        self.id_to_label: Dict[str, int] = {}
        with open(training_labels_file(data_dir)) as id_label_file:
            for id_label in id_label_file:
                _id, label = id_label.split(",")
                if _id != "id":
                    self.ids.append(_id)
                    self.id_to_label[_id] = int(label)
                    if n and len(self.ids) >= n:
                        break
        preprocessor_name = preprocessors[0].name
        self.data_name = (
            preprocessor_name
            if train_file(data_dir, self.ids[0], preprocessor_name).exists()
            else None
        )

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        _id = self.ids[idx]
        fpath = str(train_file(self.data_dir, _id, self.data_name))

        x = self.transform(np.load(fpath))
        y = self.target_transform(self.id_to_label[_id])

        return x, y

@dataclass
class MyDatasets:
    gw: GwDataset
    train: Subset[Tuple[Tensor, Tensor]]
    validation: Subset[Tuple[Tensor, Tensor]]
    test: GwSubmissionDataset


def gw_train_and_test_datasets(
    data_dir: Path,
    n: Optional[int],
    preprocessors: List[PreprocessorMeta],
    dtype: torch.dtype,
    device: torch.device,
) -> MyDatasets:
    def transform(x: np.ndarray) -> torch.Tensor:
        return torch.tensor(x, dtype=dtype, device=device)

    def target_transform(y: int) -> torch.Tensor:
        return torch.tensor((y,), dtype=dtype, device=device)

    gw = GwDataset(
        data_dir,
        n,
        preprocessors,
        transform=transform,
        target_transform=target_transform,
    )
    gw_test = GwSubmissionDataset(data_dir, preprocessors, transform=transform)
    num_examples = len(gw)
    num_train_examples = int(num_examples * 0.8)
    num_validation_examples = num_examples - num_train_examples
    train, validation = random_split(gw, [num_train_examples, num_validation_examples])
    return MyDatasets(gw, train, validation, gw_test)


TRAIN_LOGGING_INTERVAL = 30
SAMPLES_TO_CHECK = 300
MAX_SAMPLES_PER_KEY = 6


class ModelManager(ABC):
    @abstractmethod
    def train(
        self,
        data_dir: Path,
        n: Optional[int],
        device: torch.device,
        hp: "HyperParameters",
        prep_test_data: Optional[int],
    ):
        pass

    def _train_epoch(
        self,
        model: nn.Module,
        loss_fn: Callable[[Tensor, Tensor], Tensor],
        dataloader: DataLoader,
        num_examples: int,
        optimizer: torch.optim.Optimizer,
    ):
        model.train()
        interval_train_loss = 0.0
        for batch_num, (X, y) in enumerate(dataloader):

            optimizer.zero_grad()

            pred = model(X)
            loss = loss_fn(pred, y)

            loss.backward()
            optimizer.step()

            interval_train_loss += loss.item()
            if batch_num % TRAIN_LOGGING_INTERVAL == 0:
                interval_batches_done = TRAIN_LOGGING_INTERVAL if batch_num > 0 else 1
                interval_loss = interval_train_loss / interval_batches_done
                num_done = (batch_num + 1) * len(X)
                print(
                    f"training loss: {interval_loss:>5f}  [{num_done:>6d}/{num_examples:>6d}]"
                )
                wandb.log(
                    {
                        "train_pred": pred.detach().cpu().numpy(),
                        "train_loss": interval_loss,
                    }
                )
                interval_train_loss = 0.0

    # noinspection PyCallingNonCallable
    def _validate(
        self,
        epoch,
        model: nn.Module,
        loss_fn: Callable[[Tensor, Tensor], Tensor],
        dataloader: DataLoader,
        num_examples: int,
    ):
        model.eval()
        num_batches = len(dataloader)
        test_loss = 0.0
        correct = 0.0
        zero_pred = 0.0
        fp = 0.0
        fn = 0.0
        tp = 0.0
        tn = 0.0
        pred_all, y_all = None, None

        with torch.no_grad():
            for X, y in dataloader:
                pred = model(X)
                loss = loss_fn(pred, y)
                if pred_all is not None:
                    pred_all = np.append(pred_all, pred.cpu().data.numpy(), axis=0)
                else:
                    pred_all = pred.cpu().data.numpy()

                if y_all is not None:
                    y_all = np.append(y_all, y.cpu().data.numpy(), axis=0)
                else:
                    y_all = y.cpu().data.numpy()

                test_loss += loss.item()

                correct += torch.sum(torch.eq(pred > 0.0, y > 0.0)).item()

                # more than a few suggests a bug
                zero_pred += torch.sum(pred == 0.0).item()

                tp += torch.sum(torch.bitwise_and(pred > 0.0, y == 1)).item()
                fp += torch.sum(torch.bitwise_and(pred > 0.0, y == 0)).item()

                tn += torch.sum(torch.bitwise_and(pred < 0.0, y == 0)).item()
                fn += torch.sum(torch.bitwise_and(pred < 0.0, y == 1)).item()

        test_loss /= num_batches
        test_accuracy = 100.0 * correct / num_examples
        auc_score = roc_auc_score(y_all, pred_all)
        print(
            f"----\ntest metrics: Accuracy: {test_accuracy:>0.1f}%, Avg loss: {test_loss:>8f} \n"
        )
        wandb.log(
            {
                "epoch": epoch,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "zero_pred": zero_pred,
                "TP": tp,
                "FP": fp,
                "TN": tn,
                "FN": fn,
                "AUC": auc_score,
                "num_examples": num_examples,
            }
        )

    def _test(
        self,
        model: nn.Module,
        gw_test: GwSubmissionDataset,
    ):
        num_test_examples = len(gw_test.ids)
        fields = ['id', 'target']
        with open("submissions.csv", "w") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(fields)
            for i in range(num_test_examples):
                _id = gw_test.ids[i]
                x = gw_test[i]
                # add batch dimension
                x = torch.unsqueeze(x, 0)
                pred = model(x)
                csvwriter.writerow([_id, pred])
        print ("Finished writing to submissions.csv!")

    def _store_the_model(self, model: nn.Module):
        """
            Dump the model as a pickle file into the local disk.
        """
        cur_time = datetime.datetime.now()
        timestamp = cur_time.strftime("%m%d%Y_%H:%M:%S")
        filename = f"{timestamp}_model.pt"
        torch.save(model.state_dict(), filename)
        print (f"Latest model has been stored as {filename}")

    def _train(
        self,
        model: nn.Module,
        device: torch.device,
        data_dir: Path,
        n: Optional[int],
        preprocessors: List[PreprocessorMeta],
        hp: "HyperParameters",
        prep_test_data: Optional[int],
    ):
        data = gw_train_and_test_datasets(data_dir, n, preprocessors, hp.dtype, device)
        model.to(device, dtype=hp.dtype)
        loss_fn = nn.BCEWithLogitsLoss()
        wandb.watch(model, criterion=loss_fn, log="all", log_freq=100)
        train_dataloader = DataLoader(data.train, batch_size=hp.batch, shuffle=True)
        validation_dataloader = DataLoader(data.validation, batch_size=hp.batch, shuffle=True)
        print(hp)
        optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr)
        for epoch in range(hp.epochs):
            print(f"---------------- Epoch {epoch + 1} ----------------")
            self._train_epoch(
                model, loss_fn, train_dataloader, len(data.train), optimizer
            )
            self._validate(epoch, model, loss_fn, validation_dataloader, len(data.validation))

        confusion_sample_indices = (
            random.sample(data.validation.indices, SAMPLES_TO_CHECK)
            if len(data.validation) > SAMPLES_TO_CHECK
            else data.validation.indices
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
        self._store_the_model(model)
        if prep_test_data and prep_test_data == 1: # we want to prepare test data
            self._test(model, data.test)

        print("Done!")


@dataclass()
class HyperParameters:
    batch: int = 64
    epochs: int = 100
    lr: float = 0.0003
    dtype: torch.dtype = torch.float32

    @property
    def manager_class(self) -> Type[ModelManager]:
        return ModelManager


def train_model(
    manager: ModelManager, data_dir: Path, n: Optional[int], hp: HyperParameters, prep_test_data: Optional[int]
):
    validate_source_dir(data_dir)

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device {device_name}")
    device = torch.device(device_name)

    manager.train(data_dir, n, device, hp, prep_test_data)
