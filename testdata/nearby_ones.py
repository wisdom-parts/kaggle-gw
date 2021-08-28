import random
from typing import Tuple

import numpy as np
import pandas as pd

from gw_data import N_SIGNALS, SIGNAL_LEN

WINDOW_LEN = 20


def label(sigs: np.ndarray, window_len: int) -> bool:
    n, m = sigs.shape
    df = pd.DataFrame(sigs.transpose())
    df = df.rolling(window_len, min_periods=1).max() > 0.0
    for i in range(n):
        for j in range(n):
            if i != j and (df[i] & df[j]).any():
                return True
    return False


def generator(n_ones_per_sig: float) -> Tuple[np.ndarray, bool]:
    result = np.zeros((N_SIGNALS, SIGNAL_LEN))
    for i in range(N_SIGNALS):
        for _ in range(int(n_ones_per_sig)):
            result[i, random.randrange(0, SIGNAL_LEN)] = 1.0
    return result, label(result, WINDOW_LEN)
