from dataclasses import dataclass
from pathlib import Path
from typing import Union, List

from datargs import arg, parse

from command_line import path_to_dir
from models import train_model, ModelManager
from models.q_cnn import QCnnHp
from models.sig_rnn import SigRnnHp


@dataclass()
class Args:
    model: Union[QCnnHp, SigRnnHp] = arg(positional=True, help="which model to train")
    sources: List[Path] = arg(
        positional=True,
        help=(
            "directory(ies) containing the input dataset(s),"
            " in the original g2net directory structure"
        ),
    )


if __name__ == "__main__":
    args = parse(Args)
    manager: ModelManager = args.model.manager_class()
    train_model(manager, args.sources, args.model)
