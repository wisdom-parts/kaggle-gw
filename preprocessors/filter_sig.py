import numpy as np
import pycbc
import pycbc.filter
import scipy.signal
from pycbc.types import TimeSeries

from gw_data import N_SIGNALS, SIGNAL_LEN

# Given that the most visible signals I have looked at
# (all of the signals?) show up in a t range of roughly (1.3, 1.8),
# we need a shorter, steeper shoulder than the default alpha=0.5.
from gw_processing import timeseries_from_signal

TUKEY_WINDOW = scipy.signal.windows.tukey(4096, alpha=0.2)


def window(sigs: np.ndarray) -> np.ndarray:
    return sigs * TUKEY_WINDOW


def bandpass_ts(ts: TimeSeries, lf: float = 35.0, hf: float = 350.0) -> TimeSeries:
    hp = pycbc.filter.highpass(ts, lf, 8)
    return pycbc.filter.lowpass_fir(hp, hf, 8)


def process_sig(sig: np.ndarray) -> np.ndarray:
    from pycbc.psd import welch, interpolate

    if sig.shape != (SIGNAL_LEN,):
        raise ValueError(f"unexpected sigs shape: {sig.shape}")

    windowed = timeseries_from_signal(window(sig))
    high = pycbc.filter.highpass(windowed, 15, 8)

    # whiten
    psd = interpolate(welch(high), 1.0 / high.duration)
    white = (high.to_frequencyseries() / psd ** 0.5).to_timeseries()

    # The above whitening process was taken straight from PyCBC's example code
    # for GW150914, but it adds huge spikes for 0.0 <= t <= 0.1.
    # Rather than sort that out yet (TODO), we tukey out the spike.
    from pycbc.strain import gate_data

    white = gate_data(white, [(0.0, 0.05, 0.05)])
    # Here's an alternative approach from the example notebook we began with.
    # It adds complexity by cropping the time axis.
    # TODO: Is this better or worse?
    # white = high.whiten(0.125, 0.125)

    bandpassed = bandpass_ts(white)

    preprocessed = np.array(bandpassed)
    return (preprocessed - np.mean(preprocessed)) / np.std(preprocessed)


def process(sigs: np.ndarray) -> np.ndarray:
    if sigs.shape != (N_SIGNALS, SIGNAL_LEN):
        raise ValueError(f"unexpected sigs shape: {sigs.shape}")
    return np.stack([process_sig(sig) for sig in sigs])
