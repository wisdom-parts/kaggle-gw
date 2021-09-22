import math
from random import random

import numpy as np
import preprocessors.correlation as corr
from gw_data import SIGNAL_LEN


def test_correlate():
    a1 = [random() for _ in range(SIGNAL_LEN)]

    a2_windows = [a1[w * 512 + w : (w + 1) * 512] + [42.0] * w for w in range(8)]
    a2 = [v for window in a2_windows for v in window]

    c = corr.correlate(np.array(a1), np.array(a2))
    assert c.shape == (2, 1, 8)

    # There's a perfect correlation available in every window.
    for w in range(8):
        assert math.isclose(c[0, 0, w], 1.0, rel_tol=1e-5, abs_tol=1e-5)

    # In each case, you have to lag y by the window number.
    for w in range(8):
        assert c[1, 0, w] == w / 21.0
