import argparse
import shutil
from typing import Callable, Mapping, Tuple, Dict

import numpy as np

from gw_util import *
from gw_util import make_data_dirs
from preprocess import filter_sig

# Yields random test data as a g2net (3, 4096) array and a boolean label.
from testdata import nearby_ones

TestDataGenerator = Callable[[float], Tuple[np.ndarray, bool]]

generators: Mapping[str, TestDataGenerator] = {
    'nearby_ones': nearby_ones.generator
}

CALIBRATION_SAMPLE_SIZE = 40


def fraction_true(generator: TestDataGenerator, calibration: float) -> float:
    trues = 0
    for _ in range(CALIBRATION_SAMPLE_SIZE):
        if generator(calibration)[1]:
            trues += 1
    return float(trues) / CALIBRATION_SAMPLE_SIZE


def calibrate(generator: TestDataGenerator) -> float:
    high = 1.0
    while fraction_true(generator, high) < 0.6:
        high *= 2
    low = 0.0
    for _ in range(20):
        mid = (high + low) / 2
        ft = fraction_true(generator, mid)
        if ft < 0.4:
            low = mid
        elif ft > 0.6:
            high = mid
        else:
            print(f"calibration={mid}")
            return mid
    raise RuntimeError("generator calibration didn't converge")


def generate(generator: TestDataGenerator, n: int, data_dir: Path):
    calibration = calibrate(generator)

    data_dir.mkdir()
    train = train_dir(data_dir)
    train.mkdir()
    make_data_dirs(train)

    with open(training_labels_file(data_dir), "w") as labels_file:
        labels_file.write("id,target\n")
        for _ in range(n):
            example_id = random_example_id()
            data, label = generator(calibration)
            np.save(str(train / relative_example_path(example_id)), data)
            labels_file.write(f"{example_id},{int(label)}\n")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "generator", help="which test-data generator to run", choices=generators.keys()
    )
    arg_parser.add_argument("n", help="how many examples to generate", type=int)
    arg_parser.add_argument(
        "dest",
        help="directory for the output dataset, in the original g2net directory structure",
        type=path_that_does_not_exist,
    )
    args = arg_parser.parse_args()
    generate(generators[args.generator], args.n, args.dest)
