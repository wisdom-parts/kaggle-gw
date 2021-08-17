from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable

import numpy as np
from torch import Tensor
from torch.utils.data import Dataset, random_split
import torch

from gw_util import training_labels_file, train_file


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
    def train(self, source: Path, device: torch.device):
        pass
