from dataclasses import dataclass
from pathlib import Path
from typing import Union, Optional

from datargs import arg, parse

from models import train_model, ModelManager
from models.q_cnn import QCnnHp
from models.q_efficient_net import QEfficientNetHP
from models.q_resnet import QResnetHp
from models.sig_cnn import SigCnnHp
from models.sig_rnn import SigRnnHp


@dataclass
class Args:
    model: Union[SigRnnHp, SigCnnHp, QCnnHp, QResnetHp, QEfficientNetHP] = arg(
        positional=True, help="which model to train"
    )
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
    prep_data_for_submission: Optional[int] = arg(
        aliases=["-ps"],
        choices=[0, 1], # 1 indicates we want to prep data and 0 otherwise.
        default=0,
        help="flag to indicate if prep data for submission",
    )


if __name__ == "__main__":
    args = parse(Args)
    manager: ModelManager = args.model.manager_class()
    train_model(manager, args.data_dir, args.n, args.model, args.prep_data_for_submission)
