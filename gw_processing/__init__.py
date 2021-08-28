from typing import List

import numpy as np
from pycbc.types import TimeSeries
from scipy.signal.windows import tukey

from gw_data import DELTA_T, N_SIGNALS


def timeseries_from_signal(sig: np.ndarray) -> TimeSeries:
    return TimeSeries(sig, epoch=0, delta_t=DELTA_T)


def timeseries_from_signals(sigs: np.ndarray) -> List[TimeSeries]:
    return [timeseries_from_signal(sigs[i]) for i in range(N_SIGNALS)]


TUKEY_WINDOW = tukey(4096, alpha=0.2)


def window_sigs(sigs: np.ndarray) -> np.ndarray:
    return sigs * TUKEY_WINDOW
