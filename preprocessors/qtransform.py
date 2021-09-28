from typing import Tuple, cast

import numpy as np
from scipy.signal.windows import tukey

from gw_data import N_SIGNALS, SIGNAL_LEN, SIGNAL_SECS
from gw_processing import timeseries_from_signal
from preprocessor_meta import qtransform3_meta

WINDOW = tukey(SIGNAL_LEN, alpha=0.1)


def qtransform_sig(
    sig: np.ndarray,
    output_shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs a qtransform on sig, returning
    * times (numpy.ndarray) – The times that the qtransform is sampled.
    * freqs (numpy.ndarray) – The frequencies that the qtransform is sampled.
    * qplane (numpy.ndarray (2d)) – The two dimensional interpolated qtransform of this time series,
    *        in the specified output_shape representing (freqs, times)
    """
    windowed = timeseries_from_signal(sig * WINDOW)
    # As of this writing, the return type for qtransform is incorrectly declared. (or inferred?)
    # noinspection PyTypeChecker
    times: np.ndarray
    freqs: np.ndarray
    qplane: np.ndarray
    times, freqs, qplane = windowed.qtransform(
        delta_t=SIGNAL_SECS / output_shape[1],
        frange=(25, 500),
        logfsteps=output_shape[0],
        return_complex=True,
    )
    return times, freqs, qplane


def process_sig(sig: np.ndarray, output_shape: Tuple[int, int]) -> np.ndarray:
    _, _, plane = qtransform_sig(sig, output_shape)
    return plane


def process_given_shape(
    sigs: np.ndarray, output_shape: Tuple[int, int, int]
) -> np.ndarray:
    if sigs.shape != (N_SIGNALS, SIGNAL_LEN):
        raise ValueError(f"unexpected sigs shape: {sigs.shape}")
    planes = [process_sig(sig, output_shape[1:]) for sig in sigs]
    real_planes = [np.real(plane) for plane in planes]
    complex_planes = [np.imag(plane) for plane in planes]
    result = np.array(real_planes + complex_planes)
    assert result.shape == output_shape
    return result


def process(sigs: np.ndarray) -> np.ndarray:
    return process_given_shape(
        sigs, cast(Tuple[int, int, int], qtransform3_meta.output_shape)
    )
