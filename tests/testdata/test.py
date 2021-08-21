import numpy as np

from testdata import nearby_ones


def test_nearby_ones_label():
    assert nearby_ones.label(
        np.array(
            [
                [0, 0, 1, 0, 0, 1],
                [1, 0, 0, 0, 0, 0],
            ]
        ),
        3,
    )
    assert not nearby_ones.label(
        np.array(
            [
                [0, 0, 1, 0, 0, 1],
                [1, 0, 0, 0, 0, 0],
            ]
        ),
        2,
    )
