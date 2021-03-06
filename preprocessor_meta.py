from dataclasses import dataclass
from enum import Enum
from math import ceil
from typing import Tuple, Dict
from gw_data import SIGNAL_SECS, N_SIGNALS, SIGNAL_LEN, SIGNAL_TIMES, SIGNAL_DELTA_T

# We separate these constants from the preprocessing module because it goes badly for
# wandb's background process to try to import pycbc.

FILTER_CROP = 128  # values to crop from each edge after filter_sig
FILTER_LEN = SIGNAL_LEN - 2 * FILTER_CROP
FILTER_SECS = FILTER_LEN * SIGNAL_DELTA_T
FILTER_TIMES = SIGNAL_TIMES[FILTER_CROP:-FILTER_CROP]

QT2_FREQ_STEPS = 64
QT2_TIME_STEPS_PER_SEC = 128
QT2_TIME_STEPS = round(SIGNAL_SECS * QT2_TIME_STEPS_PER_SEC)
QT2_OUTPUT_SHAPE = (N_SIGNALS, QT2_FREQ_STEPS, QT2_TIME_STEPS)

QT3_FREQ_STEPS = 128
QT3_TIME_STEPS_PER_SEC = 64
QT3_TIME_STEPS = round(SIGNAL_SECS * QT3_TIME_STEPS_PER_SEC)
QT3_OUTPUT_SHAPE = (N_SIGNALS, QT3_FREQ_STEPS, QT3_TIME_STEPS)

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
qtransform2_meta = PreprocessorMeta(
    "qtransform", 2, (N_SIGNALS, QT2_FREQ_STEPS, QT2_TIME_STEPS)
)
qtransform3_meta = PreprocessorMeta(
    "qtransform", 3, (N_SIGNALS * 2, QT3_FREQ_STEPS, QT3_TIME_STEPS)
)
correlation_meta = PreprocessorMeta(
    "correlation", 2, (CORR_NUM_LAGS, 2, CORR_NUM_WINDOWS)
)


class Preprocessor(Enum):
    RAW = raw_meta
    FILTER_SIG = filter_sig_meta
    QTRANSFORM2 = qtransform2_meta
    QTRANSFORM3 = qtransform3_meta
    CORRELATION = correlation_meta


_meta_by_processor_name: Dict[str, PreprocessorMeta] = dict(
    [(e.value.name, e.value) for e in Preprocessor]
)


def meta_for_processor_name(name: str) -> PreprocessorMeta:
    meta = _meta_by_processor_name.get(name)
    if meta:
        return meta
    else:
        raise ValueError(f'unknown preprocessor name "{name}"')
