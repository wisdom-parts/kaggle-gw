import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
from pycbc.types import TimeSeries, FrequencySeries

from command_line import path_to_dir
from gw_data import (
    train_file,
    training_labels_file,
    FREQ_SERIES_DELTA_F,
    NOISE_FILENAME,
    N_SIGNALS,
    SIGNAL_DELTA_T,
)
from preprocessors import filter_sig


def update_psd(
    det_psd: FrequencySeries, fs: FrequencySeries, count: int
) -> FrequencySeries:
    return (abs(fs) + (count - 1) * det_psd) / count


def sig_to_fs(sig: np.ndarray) -> FrequencySeries:
    ts = TimeSeries(sig * filter_sig.WINDOW, delta_t=SIGNAL_DELTA_T)
    return ts.to_frequencyseries(FREQ_SERIES_DELTA_F)


def compute_noise(source_dir: Path, sample_ids: List[str]) -> np.ndarray:
    if len(sample_ids) == 0:
        raise ValueError("sample_ids cannot be empty")

    # Algorithm provided in https://github.com/gwastro/pycbc/issues/3761#issuecomment-895066248
    psds: Optional[List[FrequencySeries]] = None

    for count, idd in enumerate(sample_ids, start=1):
        sigs = np.load(str(train_file(source_dir, idd)))
        fss = [sig_to_fs(sigs[i]) for i in range(N_SIGNALS)]
        psds = [
            (update_psd(psds[i], fss[i], count) if psds else abs(fss[i]))
            for i in range(N_SIGNALS)
        ]
        if count % 1000 == 0:
            print(f"Completed {count} of {len(sample_ids)}")

    assert psds
    return np.stack(psds)


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
