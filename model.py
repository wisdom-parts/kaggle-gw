import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Union, Optional

from datargs import arg, parse

from gw_data import sample_submission_file
from models import train_model, ModelManager
from models.kitchen_sink import KitchenSinkHp
from models.q_cnn import QCnnHp
from models.q_cnn2 import QCnn2Hp
from models.q_efficient_net import QEfficientNetHp
from models.q_resnet import QResnetHp
from models.cnn1d import Cnn1dHp
from models.sig_rnn import SigRnnHp
from models.cnn1d_w_stft import Cnn1dSTFTHp


@dataclass
class Args:
    model: Union[
        SigRnnHp, Cnn1dHp, QCnnHp, QCnn2Hp, QResnetHp, QEfficientNetHp, KitchenSinkHp, Cnn1dSTFTHp
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
    submission: int = arg(
        aliases=["-s"],
        default=0,
        help="flag to indicate if prep data for submission",
    )


if __name__ == "__main__":
    args = parse(Args)
    if args.submission and not sample_submission_file(args.data_dir).exists():
        print(
            f"--submission flag passed, but no sample submission file in {args.data_dir}",
            file=sys.stderr,
        )
        sys.exit(1)
    manager: ModelManager = args.model.manager_class()
    train_model(manager, args.data_dir, args.n, args.model, args.submission)
