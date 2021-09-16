import argparse
import datetime
import re
import shutil
import sys
from typing import Callable, Mapping, Set

from command_line import path_to_dir
from gw_data import *
from gw_data import make_data_dirs
from preprocessors import filter_sig, qtransform, correlation

ProcessFunction = Callable[[np.ndarray], np.ndarray]

process_fns: Mapping[str, ProcessFunction] = {
    "filter_sig": filter_sig.process,
    "qtransform": qtransform.process,
    "correlation": correlation.process,
    "cp": lambda x: x,
}


def preprocess_train_or_test(
    process_fn: ProcessFunction,
    source: Path,
    source_data_name: Optional[str],
    dest: Path,
    dest_data_name: str,
    ids_to_process: Set[str],
):
    print("======================")
    print(f"Preprocessing {source} -> {dest}")

    make_data_dirs(dest)

    print(f"{len(ids_to_process)} rows to process")

    count = 0
    for example_id in ids_to_process:
        source_path = relative_example_path(example_id, source_data_name)
        x = np.load(str(source / source_path))
        dest_path = relative_example_path(example_id, dest_data_name)
        np.save(str(dest / dest_path), process_fn(x))
        count += 1
        if count % 100 == 0:
            print(f"{datetime.datetime.now()}: processed {count} rows")

    print("Done!")


def preprocess(
    process_fn: ProcessFunction,
    source: Path,
    source_data_name: Optional[str],
    dest: Path,
    dest_data_name: str,
    num_train_examples: Optional[int],
):
    has_test_data = validate_source_dir(source)

    os.makedirs(dest, exist_ok=True)

    all_ids = read_first_column(training_labels_file(source))

    num_ids = num_train_examples or len(all_ids)
    chosen_ids = set(all_ids[0:num_ids])

    if training_labels_file(dest).exists():
        existing_ids = set(read_first_column(training_labels_file(dest)))
        if chosen_ids != existing_ids:
            print(
                f"Tried to process a different set of training example ids than already exists in {dest}",
                file=sys.stderr,
            )
            sys.exit(-1)
    else:
        with open(training_labels_file(source)) as training_labels_in:
            with open(training_labels_file(dest), "w") as training_labels_out:
                for line in training_labels_in:
                    example_id = line.split(",")[0]
                    if example_id == "id" or example_id in chosen_ids:
                        training_labels_out.write(line)

    preprocess_train_or_test(
        process_fn,
        source=train_dir(source),
        source_data_name=source_data_name,
        dest=train_dir(dest),
        dest_data_name=dest_data_name,
        ids_to_process=chosen_ids,
    )

    if has_test_data and not num_train_examples:
        dest_sample_submission = sample_submission_file(dest)
        if not dest_sample_submission.exists():
            shutil.copy(sample_submission_file(source), dest_sample_submission)

        all_ids = read_first_column(sample_submission_file(source))
        preprocess_train_or_test(
            process_fn,
            source=test_dir(source),
            source_data_name=source_data_name,
            dest=test_dir(dest),
            dest_data_name=dest_data_name,
            ids_to_process=set(all_ids),
        )


def read_first_column(path: Path) -> List[str]:
    vs: List[str] = []
    with open(path) as file:
        it = file
        next(it, None)  # skip header row
        for line in it:
            vs.append(line.split(",")[0])
    return vs


class DataPartition:
    def __init__(self, spec: str):
        if not re.fullmatch(r"\d+/\d+", spec):
            raise ValueError("must specify data partition as I/N")
        i_str, n_str = spec.split("/")

        self.partition = int(i_str)
        self.num_partitions = int(n_str)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "-n",
        help="number of training examples to preprocess (if set, test examples are omitted)",
        type=int,
    )
    arg_parser.add_argument(
        "--from",
        dest="source_data_name",
        help="source data name (just for cp, otherwise ignored)",
    )
    arg_parser.add_argument(
        "--to",
        dest="dest_data_name",
        help="destination data name (just for cp, otherwise ignored)",
    )
    arg_parser.add_argument(
        "--partition",
        help='which partition to process, where "1/6" means partition 1 of 6',
        type=DataPartition,
    )
    arg_parser.add_argument(
        "processor", help="which processor to run", choices=process_fns.keys()
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
    source_data_name = (
        args.source_data_name
        if args.processor == "cp"
        else ("filter_sig" if args.processor in ("correlation", "qtransform") else None)
    )
    preprocess(
        process_fns[args.processor],
        args.source,
        source_data_name,
        args.dest,
        args.dest_data_name if args.processor == "cp" else args.processor,
        args.n,
    )
