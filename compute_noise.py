import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
from pycbc.types import TimeSeries, FrequencySeries
from scipy.signal.windows import tukey

from command_line import path_to_dir
from gw_data import (
    train_file,
    training_labels_file,
    FREQ_SERIES_DELTA_F,
    NOISE_FILENAME,
    N_SIGNALS,
    SIGNAL_DELTA_T,
)
from preprocessors.filter_sig import WINDOW


def update_psd(
    det_psd: FrequencySeries, fs: FrequencySeries, count: int
) -> FrequencySeries:
    if not det_psd:
        return abs(fs)
    else:
        return (abs(fs) + (count - 1) * det_psd) / count


def sig_to_fs(sig: np.ndarray) -> FrequencySeries:
    ts = WINDOW * TimeSeries(sig, delta_t=SIGNAL_DELTA_T)
    return ts.to_frequencyseries(FREQ_SERIES_DELTA_F)


def compute_noise(source_dir: Path, sample_ids: List[str]) -> np.ndarray:
    # Algorithm provided in https://github.com/gwastro/pycbc/issues/3761#issuecomment-895066248

    psd: Optional[List[FrequencySeries]] = None

    for count, idd in enumerate(sample_ids, start=1):
        sigs = np.load(str(train_file(source_dir, idd)))
        fs = [sig_to_fs(sigs[i]) for i in range(N_SIGNALS)]
        psd = (
            [update_psd(psd[i], fs[i], count) for i in range(N_SIGNALS)] if psd else fs
        )
        if count % 1000 == 0:
            print(f"Completed {count} of {len(sample_ids)}")

    if not psd:
        raise ValueError("no data")
    return np.stack(psd)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "source",
        help="directory containing the input dataset, in the original g2net directory structure",
        type=path_to_dir,
    )
    arg_parser.add_argument(
        "dest",
        help="directory to write noise.npy",
        type=Path,
    )
    args = arg_parser.parse_args()

    rows = np.loadtxt(str(training_labels_file(args.source)), delimiter=",", dtype=str)
    negative_ids = [idd for idd, label in rows[1:] if label == "0"]

    noise = compute_noise(args.source, negative_ids)
    np.save(args.dest / NOISE_FILENAME, noise)
