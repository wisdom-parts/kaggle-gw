from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from torch import Tensor
from torch.utils.data import Dataset
import torch

from gw_util import training_labels_file, train_file


class GwDataset(Dataset):
    """
    Represents the training examples of a g2net data directory as float32 Tensor's
    of size (N_SIGNALS, SIGNAL_LEN).
    """
    def __init__(self, source: Path):
        self.source = source
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

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        _id = self.ids[idx]
        fpath = str(train_file(self.source, _id))
        return torch.tensor(np.load(fpath), dtype=torch.float32), self.id_to_label[_id]


class ModelManager(ABC):
    @abstractmethod
    def train(self,
              train_dataset: GwDataset,
              test_dataset: GwDataset,
              device: torch.device):
        pass
