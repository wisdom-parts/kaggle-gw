import argparse
from typing import Type, Mapping

from gw_util import *
from gw_util import validate_source_dir

from model import ModelManager, GwDataset
from model import rnn

import torch

models: Mapping[str, Type[ModelManager]] = {"rnn": rnn.RnnManager}


def train_model(manager: ModelManager, source: Path):
    validate_source_dir(source)

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device {device_name}")
    device = torch.device(device_name)

    manager.train(GwDataset(source), device)


def existing_dir_path(s: str) -> Path:
    if os.path.isdir(s):
        return Path(s)
    else:
        raise NotADirectoryError(s)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("model", help="which model to train", choices=models.keys())
    arg_parser.add_argument(
        "source",
        help="directory containing the input dataset, in the original g2net directory structure",
        type=existing_dir_path,
    )
    args = arg_parser.parse_args()
    model_class = models[args.model]
    model = model_class()
    train_model(model, args.source)