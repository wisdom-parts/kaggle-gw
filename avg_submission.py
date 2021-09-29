import argparse
import numpy as np
import sys
from pathlib import Path
from typing import List

import pandas as pd


def sigmoid(x: pd.Series) -> pd.Series:
    return 1.0 / (1.0 + np.exp(-x))


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "files",
        nargs="+",
        help="paths of submission files to average together",
        type=Path,
    )
    args = arg_parser.parse_args()
    total_df = None
    for path in args.files:
        df = pd.read_csv(
            path, dtype={"id": "string", "target": "float"}, index_col="id"
        )
        df.sort_index(inplace=True)
        s = sigmoid(df["target"])
        if total_df is None:
            total_df = pd.DataFrame({"total": s})
        else:
            if not df.index.equals(total_df.index):
                print(f"{path} has different ids than previous files", file=sys.stderr)
                exit(1)
            total_df = total_df.assign(total=total_df["total"] + s)
    result_df = pd.DataFrame({"target": total_df["total"] / len(args.files)})
    # We are getting pandas type definitions that insist the path must be None
    # noinspection PyTypeChecker
    result_df.to_csv("avg_submission.csv")


if __name__ == "__main__":
    main()
