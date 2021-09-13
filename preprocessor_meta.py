from dataclasses import dataclass
from enum import Enum
from typing import Tuple
from gw_data import SIGNAL_SECS, N_SIGNALS, SIGNAL_LEN

# We separate these constants from the preprocessing module because it goes badly for
# wandb's background process to try to import pycbc.

FREQ_STEPS = 32
TIME_STEPS_PER_SEC = 64
TIME_STEPS = round(SIGNAL_SECS * TIME_STEPS_PER_SEC)
OUTPUT_SHAPE = (N_SIGNALS, FREQ_STEPS, TIME_STEPS)
FREQ_STEPS_64x256 = 64
TIME_STEPS_PER_SEC_64x256 = 128
TIME_STEPS_64x256 = round(SIGNAL_SECS * TIME_STEPS_PER_SEC_64x256)
OUTPUT_SHAPE_64x256 = (N_SIGNALS, FREQ_STEPS_64x256, TIME_STEPS_64x256)


@dataclass
class PreprocessorMeta:
    name: str
    output_shape: Tuple[int, ...]


filter_sig_meta = PreprocessorMeta("filter_sig", (N_SIGNALS, SIGNAL_LEN))
qtransform_meta = PreprocessorMeta("qtransform", (N_SIGNALS, 32, 128))
qtransform_64x256_meta = PreprocessorMeta("qtransform_64x256", (N_SIGNALS, 64, 256))


class Preprocessor(Enum):
    FILTER_SIG = filter_sig_meta
    QTRANSFORM = qtransform_meta
    QTRANSFORM_64X256 = qtransform_64x256_meta
