import numpy as np

from gw_util import N_SIGNALS, SIGNAL_LEN, qtransform_sig


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
