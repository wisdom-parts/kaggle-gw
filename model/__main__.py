import argparse
from typing import Type, Mapping

import torch
import wandb

from gw_util import *
from gw_util import validate_source_dir
from model import ModelManager
from model import rnn
from model import cnn_dean

models: Mapping[str, Type[ModelManager]] = {
    "rnn": rnn.Manager,
    "cnn_dean": cnn_dean.Manager,
}


def train_model(manager: ModelManager, source: Path):
    validate_source_dir(source)

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device {device_name}")
    device = torch.device(device_name)

    manager.train(source, device)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("model", help="which model to train", choices=models.keys())
    arg_parser.add_argument(
        "source",
        help="directory containing the input dataset, in the original g2net directory structure",
        type=path_to_dir,
    )
    args = arg_parser.parse_args()

    wandb.init(project="g2net-" + args.model)

    model_class = models[args.model]
    model = model_class()
    train_model(model, args.source)
