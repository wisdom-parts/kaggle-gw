import argparse
from typing import Type, Mapping

from gw_util import *
from gw_util import validate_source_dir

from model import Model
from model import rnn


def existing_dir_path(s: str) -> Path:
    if os.path.isdir(s):
        return Path(s)
    else:
        raise NotADirectoryError(s)


models: Mapping[str, Type[Model]] = {"rnn": rnn.RnnModel}


def train_model(model: Model, source: Path):
    validate_source_dir(source)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "model", help="which model to train", choices=models.keys()
    )
    arg_parser.add_argument(
        "source",
        help="directory containing the input dataset, in the original g2net directory structure",
        type=existing_dir_path,
    )
    args = arg_parser.parse_args()
    model_class = models[args.model]
    model = model_class()
    train_model(model, args.source)
