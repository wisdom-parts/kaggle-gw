import argparse
from string import hexdigits

import numpy as np
import os
from pathlib import Path
import shutil
from typing import Callable, Mapping

from preprocess import filter_sig
from gw_util import *


ProcessFunction = Callable[[np.ndarray], np.ndarray]

processors: Mapping[str, ProcessFunction] = {"filter_sig": filter_sig.process}


def existing_dir_path(s: str) -> Path:
    if os.path.isdir(s):
        return Path(s)
    else:
        raise NotADirectoryError(s)


def preprocess_train_or_test(
    processor: ProcessFunction, ids_file: Path, source: Path, dest: Path
):
    hex_low = hexdigits[0:16]
    for i in hex_low:
        for j in hex_low:
            for k in hex_low:
                (dest / i / j / k).mkdir(parents=True)
    with open(ids_file) as id_rows:
        skipped_header = False
        for row in id_rows:
            if not skipped_header:
                skipped_header = True
            else:
                example_id = row.split(",")[0]
                example_path = relative_example_path(example_id)
                x = np.load(str(source / example_path))
                np.save(str(dest / example_path), processor(x))


def preprocess(processor: ProcessFunction, source: Path, dest: Path):
    if not os.path.isdir(source):
        raise NotADirectoryError(f"source directory doesn't exist: {source}")
    if os.path.exists(dest):
        raise FileExistsError(f"destination directory already exists: {dest}")
    if not os.path.isfile(sample_submission_file(source)):
        raise FileNotFoundError(f"missing {SAMPLE_SUBMISSION_FILENAME} in {source}")
    if not os.path.isfile(training_labels_file(source)):
        raise FileNotFoundError(f"missing {TRAINING_LABELS_FILENAME} in {source}")
    if not os.path.isdir(train_dir(source)):
        raise NotADirectoryError(f"missing {TRAIN_DIRNAME} in {source}")
    if not os.path.isdir(test_dir(source)):
        raise NotADirectoryError(f"missing {TEST_DIRNAME} in {source}")

    os.makedirs(dest, exist_ok=True)

    shutil.copy(sample_submission_file(source), sample_submission_file(dest))
    shutil.copy(training_labels_file(source), training_labels_file(dest))

    preprocess_train_or_test(
        processor,
        training_labels_file(source),
        source=train_dir(source),
        dest=train_dir(dest),
    )

    preprocess_train_or_test(
        processor,
        sample_submission_file(source),
        source=test_dir(source),
        dest=test_dir(dest),
    )


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "processor", help="which processor to run", choices=processors.keys()
    )
    arg_parser.add_argument(
        "source",
        help="directory containing the input dataset, in the original g2net directory structure",
        type=existing_dir_path,
    )
    arg_parser.add_argument(
        "dest",
        help="directory for the output dataset, in the original g2net directory structure",
        type=Path,
    )
    args = arg_parser.parse_args()
    preprocess(processors[args.processor], args.source, args.dest)
