import numpy as np
from nnlib.layers import Tanh
from nnlib.tensor import Tensor

import pytest


@pytest.mark.parametrize("xx", [
    ([0]),
    ([0,1]),
    ([[0,1],[2,3]])
])

def test_tanh(xx):
    tan = Tanh()
    x = np.array(xx)
    out = tan.forawar(x)

    assert np.allclose(out, np.tanh(x))
