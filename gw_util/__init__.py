import os
from pathlib import Path

N_SIGNALS = 3
SIGNAL_NAMES = ["LIGO Hanford", "LIGO Livingston", "Virgo"]
SIGNAL_LEN = 4096
SIGNAL_SECONDS = 2.0
DELTA_T = SIGNAL_SECONDS / SIGNAL_LEN
SIGNAL_TIMES = [i * DELTA_T for i in range(SIGNAL_LEN)]

SAMPLE_SUBMISSION_FILENAME = "sample_submission.csv"
TRAINING_LABELS_FILENAME = "training_labels.csv"
TRAIN_DIRNAME = "train"
TEST_DIRNAME = "test"


def sample_submission_file(data_dir: Path) -> Path:
    return data_dir / SAMPLE_SUBMISSION_FILENAME


def training_labels_file(data_dir: Path) -> Path:
    return data_dir / TRAINING_LABELS_FILENAME


def train_dir(data_dir: Path) -> Path:
    return data_dir / TRAIN_DIRNAME


def relative_example_path(example_id: str):
    return Path(example_id[0]) / example_id[1] / example_id[2] / (example_id + ".npy")


def train_file(data_dir: Path, example_id: str) -> Path:
    return train_dir(data_dir) / relative_example_path(example_id)


def test_dir(data_dir: Path) -> Path:
    return data_dir / TEST_DIRNAME


def test_file(data_dir: Path, example_id: str) -> Path:
    return test_dir(data_dir) / relative_example_path(example_id)


def validate_source_dir(source_dir: Path) -> bool:
    """
    Validates source_dir as containing at least training data and maybe test data.

    :return: True if source_dir contains test data.
    """
    if not os.path.isdir(source_dir):
        raise NotADirectoryError(f"source directory doesn't exist: {source_dir}")
    if not os.path.isfile(training_labels_file(source_dir)):
        raise FileNotFoundError(f"missing {TRAINING_LABELS_FILENAME} in {source_dir}")
    if not os.path.isdir(train_dir(source_dir)):
        raise NotADirectoryError(f"missing {TRAIN_DIRNAME} in {source_dir}")
    if not os.path.isdir(test_dir(source_dir)):
        return False
    else:
        if not os.path.isfile(sample_submission_file(source_dir)):
            raise FileNotFoundError(f"missing {SAMPLE_SUBMISSION_FILENAME} in {source_dir}")
        return True
