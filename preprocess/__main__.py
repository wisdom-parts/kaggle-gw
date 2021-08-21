import argparse
import shutil
import datetime
from typing import Callable, Mapping

from gw_util import *
from gw_util import make_data_dirs
from preprocess import filter_sig, qtransform

ProcessFunction = Callable[[np.ndarray], np.ndarray]

processors: Mapping[str, ProcessFunction] = {
    "filter_sig": filter_sig.process,
    "qtransform": qtransform.process,
}


def preprocess_train_or_test(
    processor: ProcessFunction, ids_file: Path, source: Path, dest: Path
):
    print("======================")
    print(f"Preprocessing {source} -> {dest}")

    make_data_dirs(dest)

    count = 0
    with open(ids_file) as id_rows:
        for row in id_rows:
            count += 1
    print(f"{count} rows to process")

    with open(ids_file) as id_rows:
        count = 0
        skipped_header = False
        for row in id_rows:
            if not skipped_header:
                skipped_header = True
            else:
                example_id = row.split(",")[0]
                example_path = relative_example_path(example_id)
                x = np.load(str(source / example_path))
                np.save(str(dest / example_path), processor(x))
            count += 1
            if count % 100 == 0:
                print(f"{datetime.datetime.now()}: processed {count} rows")

    print("Done!")


def preprocess(processor: ProcessFunction, source: Path, dest: Path):
    has_test_data = validate_source_dir(source)

    os.makedirs(dest, exist_ok=True)

    shutil.copy(training_labels_file(source), training_labels_file(dest))

    preprocess_train_or_test(
        processor,
        training_labels_file(source),
        source=train_dir(source),
        dest=train_dir(dest),
    )

    if has_test_data:
        shutil.copy(sample_submission_file(source), sample_submission_file(dest))
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
        type=path_to_dir,
    )
    arg_parser.add_argument(
        "dest",
        help="directory for the output dataset, in the original g2net directory structure",
        type=Path,
    )
    args = arg_parser.parse_args()
    preprocess(processors[args.processor], args.source, args.dest)
