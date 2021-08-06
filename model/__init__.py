from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from torch.utils.data import Dataset
import torch

from gw_util import training_labels_file, train_file


class GwDataset(Dataset):
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

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        _id = self.ids[idx]
        fpath = str(train_file(self.source, _id))
        return np.load(fpath), self.id_to_label[_id]


class ModelManager(ABC):
    @abstractmethod
    def train(self, dataset: GwDataset, device: torch.device):
        pass
