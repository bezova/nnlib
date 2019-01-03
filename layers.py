from nnlib.tensor import Tensor
import numpy as np
from typing import Dict

class Layer():
    def __init__(self) -> None:
        params: Dict[str, Tensor] = {}
        pass
    
    def forward(self, input_dat: Tensor) -> Tensor:
        raise NotImplementedError

    def backward(self, input_data: Tensor) -> Tensor:
        raise NotImplementedError

class Linear(Layer):
    def __init__(self, inp: int, out: int) -> None:
        super().__init__()
        
        params['w'] = np.random.randn(inp, out)
        params['b'] = np.random.randn(out)
    
    def forward(self, input_dat: Tensor) -> Tensor:
        self.inp_dat = input_dat
        return input_dat @ params['w'] + params['b']

    def backward(self, input_data: Tensor) -> Tensor:
        pass