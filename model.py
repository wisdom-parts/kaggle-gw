from dataclasses import dataclass
from pathlib import Path
from typing import Union, Optional

from datargs import arg, parse

from models import train_model, ModelManager
from models.kitchen_sink import KitchenSinkHp
from models.q_cnn import QCnnHp
from models.q_efficient_net import QEfficientNetHp
from models.q_resnet import QResnetHp
from models.sig_cnn import SigCnnHp
from models.sig_rnn import SigRnnHp


@dataclass
class Args:
    model: Union[
        SigRnnHp, SigCnnHp, QCnnHp, QResnetHp, QEfficientNetHp, KitchenSinkHp
    ] = arg(positional=True, help="which model to train")
    data_dir: Path = arg(
        positional=True,
        help=(
            "directory containing the input dataset(s),"
            " in the original g2net directory structure"
        ),
    )
    n: Optional[int] = arg(
        aliases=["-n"],
        default=None,
        help="number of training examples to use",
    )


if __name__ == "__main__":
    args = parse(Args)
    manager: ModelManager = args.model.manager_class()
    train_model(manager, args.data_dir, args.n, args.model)
