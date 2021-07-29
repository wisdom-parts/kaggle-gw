import argparse
import enum
import os
from pathlib import Path
import shutil
from typing import Callable, Mapping

from .filter_sig import filter_sig

SAMPLE_SUBMISSION_FILENAME = 'sample_submission.csv'
TRAINING_LABELS_FILENAME = 'training_labels.csv'
TRAIN_DIRNAME = 'train'
TEST_DIRNAME = 'test'


def sample_submission_file(data_dir: Path) -> Path:
    return data_dir / SAMPLE_SUBMISSION_FILENAME


def training_labels_file(data_dir: Path) -> Path:
    return data_dir / TRAINING_LABELS_FILENAME


def train_dir(data_dir: Path) -> Path:
    return data_dir / TRAIN_DIRNAME


def test_dir(data_dir: Path) -> Path:
    return data_dir / TEST_DIRNAME


ProcessorFunType = Callable[[Path, Path, bool], None]

processors: Mapping[str, ProcessorFunType] = {
    'filter_sig': filter_sig
}


def existing_dir_path(s: str) -> Path:
    if os.path.isdir(s):
        return Path(s)
    else:
        raise NotADirectoryError(s)


def preprocess(processor: ProcessorFunType,
               source: Path, dest: Path):
    if not os.path.isdir(source):
        raise NotADirectoryError(f'source directory doesn\'t exist: {source}')
    if os.path.exists(dest):
        raise FileExistsError(f'destination directory already exists: {dest}')
    if not os.path.isfile(sample_submission_file(source)):
        raise FileNotFoundError(f'missing {SAMPLE_SUBMISSION_FILENAME} in {source}')
    if not os.path.isfile(training_labels_file(source)):
        raise FileNotFoundError(f'missing {TRAINING_LABELS_FILENAME} in {source}')
    if not os.path.isdir(train_dir(source)):
        raise NotADirectoryError(f'missing {TRAIN_DIRNAME} in {source}')
    if not os.path.isdir(test_dir(source)):
        raise NotADirectoryError(f'missing {TEST_DIRNAME} in {source}')

    os.makedirs(dest, exist_ok=True)

    shutil.copy(sample_submission_file(source),
                sample_submission_file(dest))
    shutil.copy(training_labels_file(source),
                training_labels_file(dest))


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('processor',
                            help='which processor to run',
                            choices=processors.keys())
    arg_parser.add_argument('source',
                            help='directory containing the input dataset, in the original g2net directory structure',
                            type=existing_dir_path)
    arg_parser.add_argument('dest',
                            help='directory for the output dataset, in the original g2net directory structure',
                            type=Path)
    args = arg_parser.parse_args()
    preprocess(processors[args.processor], args.source, args.dest)
