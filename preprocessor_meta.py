from dataclasses import dataclass
from enum import Enum
from math import ceil
from typing import Tuple
from gw_data import SIGNAL_SECS, N_SIGNALS, SIGNAL_LEN

# We separate these constants from the preprocessing module because it goes badly for
# wandb's background process to try to import pycbc.

FREQ_STEPS = 64
TIME_STEPS_PER_SEC = 128
TIME_STEPS = round(SIGNAL_SECS * TIME_STEPS_PER_SEC)
OUTPUT_SHAPE = (N_SIGNALS, FREQ_STEPS, TIME_STEPS)

CORR_NUM_WINDOWS = 16
CORR_WINDOW_LEN = SIGNAL_LEN // CORR_NUM_WINDOWS
CORR_MAX_LAG_MILLIS = 10
CORR_MAX_LAG = ceil(CORR_MAX_LAG_MILLIS * SIGNAL_LEN / (1000 * SIGNAL_SECS))
CORR_NUM_LAGS = 2 * CORR_MAX_LAG + 1


@dataclass
class PreprocessorMeta:
    name: str
    output_shape: Tuple[int, ...]


filter_sig_meta = PreprocessorMeta("filter_sig", (N_SIGNALS, SIGNAL_LEN))
qtransform_meta = PreprocessorMeta("qtransform", (N_SIGNALS, FREQ_STEPS, TIME_STEPS))
correlation_meta = PreprocessorMeta("correlation", (CORR_NUM_LAGS, 2, CORR_NUM_WINDOWS))


class Preprocessor(Enum):
    FILTER_SIG = filter_sig_meta
    QTRANSFORM = qtransform_meta
    CORRELATION = correlation_meta
