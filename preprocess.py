import argparse
import datetime
import re
import shutil
import sys
from abc import ABC, abstractmethod
from os.path import samefile
from time import sleep
from typing import Callable, Mapping, Set, Tuple, cast

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


class DataSubset(ABC):
    def __init__(self, covers_all_training_data: bool, process_test_data: bool):
        self.covers_all_training_data = covers_all_training_data
        self.process_test_data = process_test_data

    def before_preprocessing(self):
        pass

    @abstractmethod
    def ids_to_process(self, source_ids: List[str]) -> Set[str]:
        pass

    def before_creating_dest(self):
        pass


class DataAll(DataSubset):
    def __init__(self):
        super().__init__(covers_all_training_data=True, process_test_data=True)

    def ids_to_process(self, source_ids: List[str]) -> Set[str]:
        return set(source_ids)


class DataN(DataSubset):
    def __init__(self, spec: str):
        super().__init__(covers_all_training_data=False, process_test_data=False)
        self.n = int(spec)

    def ids_to_process(self, source_ids: List[str]) -> Set[str]:
        if self.n > len(source_ids):
            raise ValueError(
                f"There are fewer than {self.n} training examples in the source directory."
            )
        return set(source_ids[0 : self.n])


class DataPartition(DataSubset):
    def __init__(self, spec: str):
        super().__init__(covers_all_training_data=True, process_test_data=True)
        if not re.fullmatch(r"\d+/\d+", spec):
            raise ValueError("must specify data partition as P/N")
        p_str, n_str = spec.split("/")

        self.partition = int(p_str)
        self.num_partitions = int(n_str)
        if self.partition < 1 or self.partition > self.num_partitions:
            raise ValueError(
                f"partition {self.partition} not between 1 and {self.num_partitions}"
            )

    def before_preprocessing(self):
        if self.partition != 1:
            # Give partition 1 time to create the destination directory and training labels file.
            sleep(10)

    def ids_to_process(self, source_ids: List[str]) -> Set[str]:
        partition_size = len(source_ids) // self.num_partitions
        start = (self.partition - 1) * partition_size
        end = (
            self.partition * partition_size
            if self.partition < self.num_partitions
            else len(source_ids)
        )
        return set(source_ids[start:end])

    def before_creating_dest(self):
        if self.partition != 1:
            raise RuntimeError(
                f"partition {self.partition} was run before partition 1 created the destination directory"
            )


def preprocess_train_or_test(
    process_fn: ProcessFunction,
    source: Path,
    source_data_name: Optional[str],
    dest: Path,
    dest_data_name: str,
    ids_to_process: Set[str],
    mean: float,
    stdev: float,
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
        raw = process_fn(x)
        normalized = (raw - mean) / stdev
        np.save(str(dest / dest_path), normalized)
        count += 1
        if count % 100 == 0:
            print(f"{datetime.datetime.now()}: processed {count} rows")

    print("Done!")


MAX_NORMALIZATION_SAMPLE_SIZE = 100
MAX_NORMALIZATION_SAMPLE_FRACTION = 0.1


def sample_mean_and_stdev(
    process_fn: ProcessFunction,
    source: Path,
    source_data_name: Optional[str],
    source_train_id_list: List[str],
) -> Tuple[float, float]:
    sample_size = min(
        MAX_NORMALIZATION_SAMPLE_SIZE,
        int(MAX_NORMALIZATION_SAMPLE_FRACTION * len(source_train_id_list)),
    )
    sample_ids = source_train_id_list[0:sample_size]
    sample_paths = [
        source / relative_example_path(idd, source_data_name) for idd in sample_ids
    ]
    sample_data = np.array([process_fn(np.load(str(path))) for path in sample_paths])
    return cast(float, np.mean(sample_data)), cast(float, np.std(sample_data))


def preprocess(
    process_fn: ProcessFunction,
    source: Path,
    source_data_name: Optional[str],
    dest: Path,
    dest_data_name: str,
    data_subset: DataSubset,
):
    has_test_data = validate_source_dir(source)

    data_subset.before_preprocessing()
    source_train_id_list = read_first_column(training_labels_file(source))
    source_train_id_set = set(source_train_id_list)
    train_ids_to_process = data_subset.ids_to_process(source_train_id_list)

    if dest.exists():
        if not data_subset.covers_all_training_data or not samefile(source, dest):
            dest_ids = set(read_first_column(training_labels_file(dest)))
            if dest_ids != (
                source_train_id_set
                if data_subset.covers_all_training_data
                else train_ids_to_process
            ):
                print(
                    f"{dest} has different training examples than the set we will process.",
                    file=sys.stderr,
                )
                sys.exit(1)
    else:
        data_subset.before_creating_dest()
        os.makedirs(dest)
        if data_subset.covers_all_training_data:
            shutil.copy(training_labels_file(source), training_labels_file(dest))
        else:
            with open(training_labels_file(source)) as training_labels_in:
                with open(training_labels_file(dest), "w") as training_labels_out:
                    for line in training_labels_in:
                        example_id = line.split(",")[0]
                        if example_id == "id" or example_id in train_ids_to_process:
                            training_labels_out.write(line)

    mean, stdev = sample_mean_and_stdev(
        process_fn, train_dir(source), source_data_name, source_train_id_list
    )

    preprocess_train_or_test(
        process_fn,
        source=train_dir(source),
        source_data_name=source_data_name,
        dest=train_dir(dest),
        dest_data_name=dest_data_name,
        ids_to_process=train_ids_to_process,
        mean=mean,
        stdev=stdev,
    )

    if has_test_data and data_subset.process_test_data:
        dest_sample_submission = sample_submission_file(dest)
        if not dest_sample_submission.exists():
            shutil.copy(sample_submission_file(source), dest_sample_submission)

        source_test_id_list = read_first_column(sample_submission_file(source))
        test_ids_to_process = data_subset.ids_to_process(source_test_id_list)
        preprocess_train_or_test(
            process_fn,
            source=test_dir(source),
            source_data_name=source_data_name,
            dest=test_dir(dest),
            dest_data_name=dest_data_name,
            ids_to_process=test_ids_to_process,
            mean=mean,
            stdev=stdev,
        )


def read_first_column(path: Path) -> List[str]:
    vs: List[str] = []
    with open(path) as file:
        it = file
        next(it, None)  # skip header row
        for line in it:
            vs.append(line.split(",")[0])
    return vs


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "-n",
        help="number of training examples to preprocess (if set, test examples are omitted)",
        type=DataN,
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

    if args.n and args.partition:
        print("can't specify both -n and --partition", file=sys.stderr)
        sys.exit(1)
    data_subset = args.n or args.partition or DataAll()

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
        data_subset,
    )


if __name__ == "__main__":
    main()
