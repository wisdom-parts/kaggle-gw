from typing import Tuple

import numpy as np

from gw_data import (
    N_SIGNALS,
    SIGNAL_LEN,
)
from gw_processing import timeseries_from_signal, window_sigs
from qtransform_params import FREQ_STEPS, TIME_STEPS_PER_SEC, OUTPUT_SHAPE


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
    result: Tuple[np.ndarray, np.ndarray, np.ndarray] = ts.qtransform(
        delta_t=1.0 / TIME_STEPS_PER_SEC, frange=(30, 350), logfsteps=FREQ_STEPS
    )
    assert result[2].shape == OUTPUT_SHAPE[1:]
    return result


def process_sig(sig: np.ndarray) -> np.ndarray:
    _, _, result = qtransform_sig(sig)
    # Normalize to (0.0 .. 1.0)
    result = result - result.min()
    result = result / result.max()
    return result


def process(sigs: np.ndarray) -> np.ndarray:
    if sigs.shape != (N_SIGNALS, SIGNAL_LEN):
        raise ValueError(f"unexpected sigs shape: {sigs.shape}")
    return np.stack([process_sig(sig) for sig in sigs])
