import numpy as np
from nnlib.layers import Tanh
from nnlib.tensor import Tensor

import pytest

'''
test typing with
mypy nnlib/  --ignore-missing-imports
'''

@pytest.mark.parametrize("xx", [
    ([0]),
    ([0,1]),
    ([[0,1],[2,3]])
])


@pytest.fixture(params=[[0], [0,2]])
def X(request):
    x = np.array(request.param)
    return  x


def test_tanh(X):
    tan = Tanh()

    out = tan.forawar(X)

    assert np.allclose(out, np.tanh(X))
