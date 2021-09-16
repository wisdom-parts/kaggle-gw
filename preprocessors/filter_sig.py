import numpy as np
import pycbc
import pycbc.filter
import scipy.signal
from pycbc.types import TimeSeries, FrequencySeries

from gw_data import N_SIGNALS, SIGNAL_LEN, NOISE_FILENAME, FREQ_SERIES_DELTA_F
from gw_processing import timeseries_from_signal

WINDOW = scipy.signal.windows.tukey(4096, alpha=0.2)


def window(sigs: np.ndarray) -> np.ndarray:
    return sigs * WINDOW


def bandpass_ts(ts: TimeSeries, lf: float = 35.0, hf: float = 350.0) -> TimeSeries:
    hp = pycbc.filter.highpass(ts, lf, 8)
    return pycbc.filter.lowpass_fir(hp, hf, 8)


def process_sig_to_ts(sig: np.ndarray, noise_psd: FrequencySeries) -> TimeSeries:
    windowed = timeseries_from_signal(window(sig))
    highpassed = pycbc.filter.highpass(windowed, 15, 8)
    whitened = (highpassed.to_frequencyseries() / noise_psd ** 0.5).to_timeseries()
    return bandpass_ts(whitened)


def process_sig(sig: np.ndarray, noise_psd: FrequencySeries) -> np.ndarray:
    if sig.shape != (SIGNAL_LEN,):
        raise ValueError(f"unexpected sig shape: {sig.shape}")
    processed_ts = process_sig_to_ts(sig, noise_psd)
    preprocessed = np.array(processed_ts)
    return (preprocessed - np.mean(preprocessed)) / np.std(preprocessed)


_noise = np.load(NOISE_FILENAME)
noise_psds = [FrequencySeries(_noise[i], FREQ_SERIES_DELTA_F) for i in range(N_SIGNALS)]


def process(sigs: np.ndarray) -> np.ndarray:
    if sigs.shape != (N_SIGNALS, SIGNAL_LEN):
        raise ValueError(f"unexpected sigs shape: {sigs.shape}")
    return np.stack([process_sig(sigs[i], noise_psds[i]) for i in range(N_SIGNALS)])
