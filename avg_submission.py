import argparse
import numpy as np
import sys
from pathlib import Path
from typing import List

import pandas as pd

EXPECTED_COLUMNS = ["id", "target"]


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "files",
        nargs="+",
        help="paths of submission files to average together",
        type=Path,
    )
    args = arg_parser.parse_args()
    dataframes: List[pd.DataFrame]
    for path in args.files:
        if not path.exists():
            print(f"file not found: {path}", file=sys.stderr)
            exit(1)
        df = pd.read_csv(path, dtype={"id": "string", "target": np.float})
        if list(df.columns) != EXPECTED_COLUMNS:
            print(
                f"each file must have columns {EXPECTED_COLUMNS}; got {list(df.columns)}",
                file=sys.stderr,
            )
            exit(1)
        dtypes = df.dtypes
        exit(0)


if __name__ == "__main__":
    main()
