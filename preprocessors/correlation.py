from typing import cast, List

import numpy as np
from numpy.linalg import norm

from preprocessor_meta import CORR_NUM_WINDOWS, CORR_WINDOW_LEN, CORR_MAX_LAG


def lagged_cosine(x: np.ndarray, y: np.ndarray, lag_y: int) -> float:
    xi0 = max(0, lag_y)
    xi1 = len(x) + min(0, lag_y)

    yi0 = max(0, -lag_y)
    yi1 = len(y) + min(0, -lag_y)

    x_slice = x[xi0:xi1]
    y_slice = y[yi0:yi1]

    x_normed = x_slice / norm(x_slice)
    y_normed = y_slice / norm(y_slice)

    dot_product = np.dot(x_normed, y_normed)
    return cast(float, dot_product)


def scan_lagged_cosine(x: np.ndarray, y: np.ndarray, max_lag: int) -> np.ndarray:
    cs = [lagged_cosine(x, y, lag_y) for lag_y in range(-max_lag, max_lag + 1)]
    return np.array(cs)


def correlate(sig1: np.ndarray, sig2: np.ndarray) -> np.ndarray:
    """
    Computes an np.ndarray shaped (NUM_LAGS, WINDOWS) with dimensions as defined for `process`
    """
    if len(sig1.shape) != 1 or sig1.shape != sig2.shape:
        raise ValueError(f"Invalid shapes: {sig1.size} and {sig2.size}")
    window_lags_list: List[np.ndarray] = []
    for window in range(CORR_NUM_WINDOWS):
        i0 = window * CORR_WINDOW_LEN
        i1 = (window + 1) * CORR_WINDOW_LEN
        window_lags_list.append(
            scan_lagged_cosine(sig1[i0:i1], sig2[i0:i1], CORR_MAX_LAG)
        )
    return np.stack(window_lags_list, axis=1)


def process(sigs: np.ndarray) -> np.ndarray:
    """
    Computes an np.ndarray of type float, shaped (NUM_LAGS, 2, WINDOWS), where

    * the indices represent (channel, signal pair, width)

    * the signal pair dimension represents comparing signals (0, 1) and then (0, 2)

    :param sigs: output from filter_sig
    """
    results = [correlate(sigs[0], sigs[i2]) for i2 in (1, 2)]
    return np.stack(results, axis=1)
