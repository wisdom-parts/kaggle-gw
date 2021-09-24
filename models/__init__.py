import csv
import datetime
import random
from abc import abstractmethod, ABC
from dataclasses import dataclass
from enum import Enum, auto
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


def to_odd(i: int) -> int:
    return (i // 2) * 2 + 1

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

    def __getitem__(self, idx: int) -> Tuple[Tensor, str]:
        _id = self.ids[idx]
        fpath = str(test_file(self.data_dir, _id, self.data_name))

        x = self.transform(np.load(fpath))
        return x, _id

class GwDataset(Dataset[Tuple[Dict[str, Tensor], Tensor]]):
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
        self.data_dir = data_dir
        self.preprocessors = preprocessors
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

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, Tensor], Tensor]:
        _id = self.ids[idx]

        xd: Dict[str, Tensor] = {}
        for preprocessor in self.preprocessors:
            fpath = str(train_file(self.data_dir, _id, preprocessor.data_name))
            v = self.transform(np.load(fpath))
            xd[preprocessor.name] = v
        y = self.target_transform(self.id_to_label[_id])
        return xd, y

@dataclass
class MyDatasets:
    gw: GwDataset
    train: Subset[Tuple[Dict[str, Tensor], Tensor]]
    validation: Subset[Tuple[Dict[str, Tensor], Tensor]]
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
    return MyDatasets(gw, train, validation, test)


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
        submission: Optional[int],
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
        for batch_num, (xd, y) in enumerate(dataloader):

            optimizer.zero_grad()

            pred = model(xd)
            loss = loss_fn(pred, y)

            loss.backward()
            optimizer.step()

            interval_train_loss += loss.item()
            if batch_num % TRAIN_LOGGING_INTERVAL == 0:
                interval_batches_done = TRAIN_LOGGING_INTERVAL if batch_num > 0 else 1
                interval_loss = interval_train_loss / interval_batches_done
                x_tensor_example = next(iter(xd.values()))
                num_done = (batch_num + 1) * len(x_tensor_example)
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

    def _test(
            self,
            model: nn.Module,
            test: GwSubmissionDataset,
            batch: int,
    ):
        fields = ['id', 'target']
        test_dataloader = DataLoader(test, batch_size=batch)
        with open("submissions.csv", "w") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(fields)
            for batch_num, (x, _id) in enumerate(test_dataloader):
                pred = model(x)
                m = torch.nn.Sigmoid()
                op = m(pred).data.cpu().numpy()[0]
                csvwriter.writerow([_id, op])
        print("Finished writing to submissions.csv!")

    def _store_the_model(self, model: nn.Module, optimizer: nn.Module, hp: "HyperParameters"):
        """
            Save the model as a state dict.
        """
        cur_time = datetime.datetime.now()
        timestamp = cur_time.strftime("%Y%d%m_%H:%M:%S")
        filename = f"model_{timestamp}.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "hp": hp.__dict__,
        }, filename)
        print (f"Latest model has been stored as {filename}")

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
        validation_loss = 0.0
        correct = 0.0
        zero_pred = 0.0
        fp = 0.0
        fn = 0.0
        tp = 0.0
        tn = 0.0
        pred_all, y_all = None, None

        with torch.no_grad():
            for xd, y in dataloader:
                pred = model(xd)
                loss = loss_fn(pred, y)
                if pred_all is not None:
                    pred_all = np.append(pred_all, pred.cpu().data.numpy(), axis=0)
                else:
                    pred_all = pred.cpu().data.numpy()

                if y_all is not None:
                    y_all = np.append(y_all, y.cpu().data.numpy(), axis=0)
                else:
                    y_all = y.cpu().data.numpy()

                validation_loss += loss.item()

                correct += torch.sum(torch.eq(pred > 0.0, y > 0.0)).item()

                # more than a few suggests a bug
                zero_pred += torch.sum(pred == 0.0).item()

                tp += torch.sum(torch.bitwise_and(pred > 0.0, y == 1)).item()
                fp += torch.sum(torch.bitwise_and(pred > 0.0, y == 0)).item()

                tn += torch.sum(torch.bitwise_and(pred < 0.0, y == 0)).item()
                fn += torch.sum(torch.bitwise_and(pred < 0.0, y == 1)).item()

        validation_loss /= num_batches
        validation_accuracy = 100.0 * correct / num_examples
        auc_score = roc_auc_score(y_all, pred_all)
        print(
            f"----\nvalidation metrics: Accuracy: {validation_accuracy:>0.1f}%, Avg loss: {validation_loss:>8f} \n"
        )
        wandb.log(
            {
                "epoch": epoch + 1,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
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
                m = torch.nn.Sigmoid()
                op = m(pred).data.cpu().numpy()[0]
                pred_val = 0 if op <= 0.5 else 1
                csvwriter.writerow([_id, pred_val])
        print ("Finished writing to submissions.csv!")

    def _store_the_model(self, model: nn.Module):
        """
            Dump the model as a pickle file into the local disk.
        """
        cur_time = datetime.datetime.now()
        timestamp = cur_time.strftime("%Y%d%m_%H:%M:%S")
        filename = f"model_{timestamp}.pt"
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
        submission: Optional[int],
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
            x = {p: torch.unsqueeze(t, 0) for p, t in x.items()}
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
        if submission == 1:  # we want to prepare test data
            self._test(model, data.test, hp.batch)

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
    manager: ModelManager, data_dir: Path, n: Optional[int], hp: HyperParameters, submission: Optional[int]
):
    validate_source_dir(data_dir)

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device {device_name}")
    device = torch.device(device_name)

    manager.train(data_dir, n, device, hp, submission)


class RegressionHead(Enum):
    LINEAR = auto()
    MAX = auto()
    AVG_LINEAR = auto()


class HpWithRegressionHead(HyperParameters):
    linear1drop: float = 0.0
    linear1out: int = 64  # if this value is 1, then omit linear2
    head: RegressionHead = RegressionHead.LINEAR


class MaxHead(nn.Module):
    """
    Consumes the output of Cnn (channel, w) or (channel, h, w) with no final activation
    and returns the maximum across all outputs for each example
    to produce a single logit with no final activation.
    """

    apply_activation_before_input = False

    # We standardize init parameters for regression heads to make it simpler to construct the one you want.
    # noinspection PyUnusedLocal
    def __init__(
        self,
        hp: HpWithRegressionHead,
        input_shape: Tuple[int, ...],
    ):
        super().__init__()

    # noinspection PyMethodMayBeStatic
    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.size()[0]
        out = torch.flatten(x, start_dim=1)
        out = torch.amax(out, dim=1, keepdim=True)
        assert out.size() == (batch_size, 1)
        return out


class LinearHead(nn.Module):
    """
    Consumes the output of Cnn (channel, w) or (channel, h, w) with a final activation and
    applies one or two linear layers to produce a single logit with no final activation.
    If hp.head == RegressionHead.AVG_LINEAR, then this module first
    takes the average across (w) or (h, w).
    """

    apply_activation_before_input = True

    # We standardize init parameters for regression heads to make it simpler to construct the one you want.
    # noinspection PyUnusedLocal
    def __init__(
        self,
        hp: HpWithRegressionHead,
        input_shape: Tuple[int, ...],
    ):
        """
        :param input_shape: (channel, w) or (channel, h, w)
        """
        super().__init__()
        if len(input_shape) not in (2, 3):
            raise ValueError(f"input shape must have length 2 or 3, was {input_shape}")

        self.hp = hp

        spacial_size = input_shape[1] * (1 if len(input_shape) == 2 else input_shape[2])

        linear_input_features = input_shape[0] * (
            1 if hp.head == RegressionHead.AVG_LINEAR else spacial_size
        )

        self.linear1 = nn.Linear(
            in_features=linear_input_features,
            out_features=hp.linear1out,
        )
        self.lin1_dropout = nn.Dropout(p=hp.linear1drop)

        self.activation = nn.ReLU()
        if hp.linear1out > 1:
            self.bn = nn.BatchNorm1d(hp.linear1out)
            self.linear2 = nn.Linear(
                in_features=hp.linear1out,
                out_features=1,
            )

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.size()[0]
        spacial_len = len(x.size()) - 2

        if self.hp.head == RegressionHead.AVG_LINEAR:
            # Average across spacial dimensions, leaving (batch, channels)
            spacial_dims = [2] if spacial_len == 1 else [2, 3]
            out = torch.mean(x, dim=spacial_dims)
        else:
            out = torch.flatten(x, start_dim=1)

        out = self.linear1(out)
        if self.hp.linear1out > 1:
            out = self.bn(out)
            out = self.activation(out)
            if self.hp.linear1drop > 0.0:
                out = self.lin1_dropout(out)
            out = self.linear2(out)

        assert out.size() == (batch_size, 1)
        return out