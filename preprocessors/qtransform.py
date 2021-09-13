from typing import Tuple, cast

import numpy as np

from gw_data import (
    N_SIGNALS,
    SIGNAL_LEN,
    SIGNAL_SECS,
)
from gw_processing import timeseries_from_signal, window_sigs
from preprocessor_meta import Preprocessor, qtransform_meta, qtransform_64x256_meta


def qtransform_sig(
    sig: np.ndarray, output_shape: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs a qtransform on sig, returning
    * times (numpy.ndarray) – The times that the qtransform is sampled.
    * freqs (numpy.ndarray) – The frequencies that the qtransform is sampled.
    * qplane (numpy.ndarray (2d)) – The two dimensional interpolated qtransform of this time series,
    *        in the specified output_shape representing (freqs, times)
    """
    ts = timeseries_from_signal(window_sigs(sig))

    # As of this writing, the return type for qtransform is incorrectly declared. (or inferred?)
    # noinspection PyTypeChecker
    result: Tuple[np.ndarray, np.ndarray, np.ndarray] = ts.qtransform(
        delta_t=SIGNAL_SECS / output_shape[1],
        frange=(30, 350),
        logfsteps=output_shape[0],
    )
    assert result[2].shape == output_shape
    return result


def process_sig(sig: np.ndarray, output_shape: Tuple[int, int]) -> np.ndarray:
    _, _, result = qtransform_sig(sig, output_shape)
    return (result - np.mean(result)) / np.std(result)


def process_given_shape(
    sigs: np.ndarray, output_shape: Tuple[int, int, int]
) -> np.ndarray:
    if sigs.shape != (N_SIGNALS, SIGNAL_LEN):
        raise ValueError(f"unexpected sigs shape: {sigs.shape}")
    return np.stack([process_sig(sig, output_shape[1:]) for sig in sigs])


def process_original(sigs: np.ndarray) -> np.ndarray:
    return process_given_shape(
        sigs, cast(Tuple[int, int, int], qtransform_meta.output_shape)
    )


def process_64x256(sigs: np.ndarray) -> np.ndarray:
    return process_given_shape(
        sigs, cast(Tuple[int, int, int], qtransform_64x256_meta.output_shape)
    )
