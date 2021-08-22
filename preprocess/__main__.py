import argparse
import collections
import shutil
import datetime
from typing import Callable, Mapping, Optional, Set
import random

from gw_util import *
from gw_util import make_data_dirs
from preprocess import filter_sig, qtransform

ProcessFunction = Callable[[np.ndarray], np.ndarray]

processors: Mapping[str, ProcessFunction] = {
    "filter_sig": filter_sig.process,
    "qtransform": qtransform.process,
}


def preprocess_train_or_test(
    processor: ProcessFunction, source: Path, dest: Path, ids_to_process: Set[str]
):
    print("======================")
    print(f"Preprocessing {source} -> {dest}")

    make_data_dirs(dest)

    print(f"{len(ids_to_process)} rows to process")

    count = 0
    for example_id in ids_to_process:
        example_path = relative_example_path(example_id)
        x = np.load(str(source / example_path))
        np.save(str(dest / example_path), processor(x))
        count += 1
        if count % 100 == 0:
            print(f"{datetime.datetime.now()}: processed {count} rows")

    print("Done!")


def preprocess(
    processor: ProcessFunction,
    source: Path,
    dest: Path,
    num_train_examples: Optional[int],
):
    has_test_data = validate_source_dir(source)

    os.makedirs(dest, exist_ok=True)

    all_ids = read_first_column(training_labels_file(source))

    chosen_ids = set(
        random.sample(all_ids, num_train_examples) if num_train_examples else all_ids
    )

    with open(training_labels_file(source)) as training_labels_in:
        with open(training_labels_file(dest), "w") as training_labels_out:
            for line in training_labels_in:
                example_id = line.split(",")[0]
                if example_id == "id" or example_id in chosen_ids:
                    training_labels_out.write(line)

    preprocess_train_or_test(
        processor,
        source=train_dir(source),
        dest=train_dir(dest),
        ids_to_process=chosen_ids,
    )

    if has_test_data and not num_train_examples:
        shutil.copy(sample_submission_file(source), sample_submission_file(dest))
        all_ids = read_first_column(sample_submission_file(source))
        preprocess_train_or_test(
            processor,
            source=test_dir(source),
            dest=test_dir(dest),
            ids_to_process=set(all_ids),
        )


def read_first_column(path: Path) -> List[str]:
    vs: List[str] = []
    with open(path) as file:
        it = file
        next(it, None) # skip header row
        for line in it:
            vs.append(line.split(",")[0])
    return vs


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "-n",
        help="number of training examples to preprocess (omits test examples)",
        type=int,
    )
    arg_parser.add_argument(
        "processor", help="which processor to run", choices=processors.keys()
    )
    arg_parser.add_argument(
        "source",
        help="directory containing the input dataset, in the original g2net directory structure",
        type=path_to_dir,
    )
    arg_parser.add_argument(
        "dest",
        help="directory for the output dataset, in the original g2net directory structure",
        type=Path,
    )
    args = arg_parser.parse_args()
    preprocess(processors[args.processor], args.source, args.dest, args.n)
