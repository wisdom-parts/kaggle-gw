import argparse
import os
from pathlib import Path


def path_to_dir(s: str) -> Path:
    if os.path.isdir(s):
        return Path(s)
    else:
        raise NotADirectoryError(s)


def path_that_does_not_exist(s: str) -> Path:
    path = Path(s)
    if path.exists():
        raise FileExistsError
    else:
        return path
