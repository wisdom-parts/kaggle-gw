from dataclasses import dataclass
from enum import Enum
from math import ceil
from typing import Tuple
from gw_data import SIGNAL_SECS, N_SIGNALS, SIGNAL_LEN, SIGNAL_TIMES, SIGNAL_DELTA_T

# We separate these constants from the preprocessing module because it goes badly for
# wandb's background process to try to import pycbc.

FILTER_CROP = 128  # values to crop from each edge after filter_sig
FILTER_LEN = SIGNAL_LEN - 2 * FILTER_CROP
FILTER_SECS = FILTER_LEN * SIGNAL_DELTA_T
FILTER_TIMES = SIGNAL_TIMES[FILTER_CROP:-FILTER_CROP]

QT_FREQ_STEPS = 128
QT_TIME_STEPS_PER_SEC = 64
QT_TIME_STEPS = round(SIGNAL_SECS * QT_TIME_STEPS_PER_SEC)
QT_OUTPUT_SHAPE = (N_SIGNALS, QT_FREQ_STEPS, QT_TIME_STEPS)

CORR_NUM_WINDOWS = 16
CORR_WINDOW_LEN = SIGNAL_LEN // CORR_NUM_WINDOWS
CORR_MAX_LAG_MILLIS = 10
CORR_MAX_LAG = ceil(CORR_MAX_LAG_MILLIS * SIGNAL_LEN / (1000 * SIGNAL_SECS))
CORR_NUM_LAGS = 2 * CORR_MAX_LAG + 1

RAW_PREPROCESSOR_NAME = "raw"


@dataclass
class PreprocessorMeta:
    name: str
    version: int
    output_shape: Tuple[int, ...]

    @property
    def data_name(self):
        if self.name == RAW_PREPROCESSOR_NAME:
            return None
        else:
            return self.name + str(self.version)


raw_meta = PreprocessorMeta(RAW_PREPROCESSOR_NAME, 1, (N_SIGNALS, SIGNAL_LEN))
filter_sig_meta = PreprocessorMeta(
    "filter_sig", 2, (N_SIGNALS, SIGNAL_LEN - 2 * FILTER_CROP)
)
qtransform_meta = PreprocessorMeta(
    "qtransform", 3, (N_SIGNALS * 2, QT_FREQ_STEPS, QT_TIME_STEPS)
)
correlation_meta = PreprocessorMeta(
    "correlation", 2, (CORR_NUM_LAGS, 2, CORR_NUM_WINDOWS)
)


class Preprocessor(Enum):
    RAW = raw_meta
    FILTER_SIG = filter_sig_meta
    QTRANSFORM = qtransform_meta
    CORRELATION = correlation_meta
