import argparse
from typing import Type, Mapping

from command_line import path_to_dir
from models import q_cnn_conventional, sig_rnn_conventional, train_model, ModelManager

models: Mapping[str, Type[ModelManager]] = {
    "q_cnn_conventional": q_cnn_conventional.Manager,
    "sig_rnn_conventional": sig_rnn_conventional.Manager,
}

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("model", help="which model to train", choices=models.keys())
    arg_parser.add_argument(
        "source",
        help="directory containing the input dataset, in the original g2net directory structure",
        type=path_to_dir,
    )

    args = arg_parser.parse_args()
    model_class = models[args.model]
    model = model_class()
    train_model(model, args.source)
