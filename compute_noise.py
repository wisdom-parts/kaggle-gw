import argparse
from pathlib import Path
from random import sample
from typing import List, Optional

from pycbc.types import TimeSeries, FrequencySeries
import numpy as np
from scipy.signal.windows import tukey

from command_line import path_to_dir
from gw_data import train_file, training_labels_file


def update_psd(
    det_psd: Optional[FrequencySeries], det_f: FrequencySeries, count: int
) -> FrequencySeries:
    if not det_psd:
        return abs(det_f)
    else:
        return (abs(det_f) + (count - 1) * det_psd) / count


def compute_noise(source_dir: Path, sample_ids: List[str]) -> np.ndarray:
    # Algorithm provided in https://github.com/gwastro/pycbc/issues/3761#issuecomment-895066248

    # Compute a window. Nothing fancy here, I just took an example from the notebooks on the Kaggle page.
    window = tukey(4096, alpha=0.2)

    det1_psd = None
    det2_psd = None
    det3_psd = None

    for count, idd in enumerate(sample_ids, start=1):
        example_arr = np.load(str(train_file(source_dir, idd)))

        det1 = TimeSeries(example_arr[0], delta_t=1.0 / 2048.0)
        det2 = TimeSeries(example_arr[1], delta_t=1.0 / 2048.0)
        det3 = TimeSeries(example_arr[2], delta_t=1.0 / 2048.0)

        det1 = det1 * window
        det2 = det2 * window
        det3 = det3 * window

        # This Fourier transforms the data
        det1f = det1.to_frequencyseries()
        det2f = det2.to_frequencyseries()
        det3f = det3.to_frequencyseries()

        # Then I take the abs value and compute the rolling mean. It would perhaps be more CPU efficient, and simpler
        # to read, to store all the `abs(detXf)` and then average at the end, but it requires a lot of RAM.
        det1_psd = update_psd(det1_psd, det1f, count)
        det2_psd = update_psd(det2_psd, det2f, count)
        det3_psd = update_psd(det3_psd, det3f, count)

        if count % 1000 == 0:
            print(f"Completed {count} of {len(sample_ids)}")

    return np.stack((det1_psd, det2_psd, det3_psd))


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

    rows = np.loadtxt(training_labels_file(args.source), delimiter=",", dtype=str)
    negative_ids = [idd for idd, label in rows[1:] if label == "0"]

    noise = compute_noise(args.source, sample(negative_ids, 100000))
    np.save(args.dest / "noise.npy")
