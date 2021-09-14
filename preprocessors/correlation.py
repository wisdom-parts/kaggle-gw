from math import ceil
from typing import cast

import numpy as np
from numpy.linalg import norm

from gw_data import SIGNAL_SECS, SIGNAL_LEN

WINDOWS = 16
WINDOW_LEN = SIGNAL_LEN // WINDOWS
MAX_LAG_MILLIS = 10
MAX_LAG = ceil(MAX_LAG_MILLIS * SIGNAL_LEN / (1000 * SIGNAL_SECS))


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
    Computes an np.ndarray shaped (2, 1, WINDOWS) with dimensions as defined for `process`
    """
    if len(sig1.shape) != 1 or sig1.shape != sig2.shape:
        raise ValueError(f"Invalid shapes: {sig1.size} and {sig2.size}")
    result = np.zeros((2, 1, WINDOWS), dtype=float)
    for window in range(WINDOWS):
        i0 = window * WINDOW_LEN
        i1 = (window + 1) * WINDOW_LEN
        cs = scan_lagged_cosine(sig1[i0:i1], sig2[i0:i1], MAX_LAG)
        result[0, 0, window] = np.max(cs)
        argmax = np.argmax(cs)
        result[1, 0, window] = (argmax - MAX_LAG) / MAX_LAG
    return result


def process(sigs: np.ndarray) -> np.ndarray:
    """
    Computes an np.ndarray of type float, shaped (2, 2, WINDOWS), where

    * the indices represent (channel, signal pair, width)

    * the signal pair dimension represents comparing signals (0, 1) and then (0, 2)

    * the two output channels are

      * a maximum (across offsets) correlation between two signals in a time window.
        (It is important that we don't normalize these values, since the magnitude of
        a correlation has intrinsic meaning. But they are between 0 and 1 anyway.)

      * an offset between -1 and 1 representing between +/- MAX_LAG_MILLIS

    :param sigs: output from filter_sig
    """
    results = [correlate(sigs[0], sigs[i2]) for i2 in (1, 2)]
    return np.stack(results, axis=1)
