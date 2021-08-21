import os
import random
from pathlib import Path
from string import hexdigits
from typing import List, Tuple

import numpy as np
import scipy.signal

from pycbc.types import TimeSeries

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
EXAMPLE_ID_LEN = 10

hex_low = hexdigits[0:16]


def random_example_id() -> str:
    return "".join([random.choice(hex_low) for _ in range(EXAMPLE_ID_LEN)])


def sample_submission_file(data_dir: Path) -> Path:
    return data_dir / SAMPLE_SUBMISSION_FILENAME


def training_labels_file(data_dir: Path) -> Path:
    return data_dir / TRAINING_LABELS_FILENAME


def train_dir(data_dir: Path) -> Path:
    return data_dir / TRAIN_DIRNAME


def relative_example_path(example_id: str):
    """
    Returns the relative path from a train or test directory to the data file for the
    given example_id.
    """
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
            raise FileNotFoundError(
                f"missing {SAMPLE_SUBMISSION_FILENAME} in {source_dir}"
            )
        return True


def timeseries_from_signal(sig: np.ndarray) -> TimeSeries:
    return TimeSeries(sig, epoch=0, delta_t=DELTA_T)


def timeseries_from_signals(sigs: np.ndarray) -> List[TimeSeries]:
    return [timeseries_from_signal(sigs[i]) for i in range(N_SIGNALS)]


TUKEY_WINDOW = scipy.signal.tukey(4096, alpha=0.2)


def window_sigs(sigs: np.ndarray) -> np.ndarray:
    return sigs * TUKEY_WINDOW


QTRANSFORM_OUTPUT_SHAPE = (30, 100)


def qtransform_sig(sig: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs a qtransform on sig, returning
    * times (numpy.ndarray) – The time that the qtransform is sampled.
    * freqs (numpy.ndarray) – The frequencies that the qtransform is sampled.
    * qplane (numpy.ndarray (2d)) – The two dimensional interpolated qtransform of this time series.
    """
    ts = timeseries_from_signal(window_sigs(sig))

    # As of this writing, the return type for qtransform is incorrectly declared. (or inferred?)
    # noinspection PyTypeChecker
    result: Tuple[np.ndarray, np.ndarray, np.ndarray] = (
        ts.qtransform(delta_t=0.02,
                      frange=(30, 350),
                      logfsteps=30))
    assert result[2].shape == QTRANSFORM_OUTPUT_SHAPE
    return result


def make_data_dirs(train_or_test_dir: Path):
    for i in hex_low:
        for j in hex_low:
            for k in hex_low:
                (train_or_test_dir / i / j / k).mkdir(parents=True)


def path_that_does_not_exist(s: str) -> Path:
    path = Path(s)
    if path.exists():
        raise FileExistsError
    else:
        return path


def path_to_dir(s: str) -> Path:
    if os.path.isdir(s):
        return Path(s)
    else:
        raise NotADirectoryError(s)
