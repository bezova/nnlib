from .tensor import Tensor
import numpy as np
from typing import Dict, Callable

class Layer():
    def __init__(self) -> None:
        self.params: Dict[str, Tensor] = {}
        self.grads: Dict[str, Tensor] = {}
    
    def forward(self, input_dat: Tensor) -> Tensor:
        raise NotImplementedError

    def backward(self, input_data: Tensor) -> Tensor:
        raise NotImplementedError


class Linear(Layer):
    def __init__(self, inp: int, out: int) -> None:
        super().__init__()        
        self.params['w'] = np.random.randn(inp, out)
        self.params['b'] = np.random.randn(out)
    
    def forward(self, input_dat: Tensor) -> Tensor:
        self.inp_dat = input_dat
        return input_dat @ self.params['w'] + self.params['b']

    def backward(self, grad: Tensor) -> Tensor:
        self.grads['b'] = np.sum(grad, axis=0)
        self.grads['w'] = self.inp_dat.T @ grad
        return grad @ self.params['w'].T


Active = Callable[[Tensor], Tensor]    


class Activation(Layer):
    def __init__(self, f: Active, f_prime: Active) -> None:
        self.f = f
        self.f_prime = f_prime

    def forawar(self, input_data: Tensor) -> Tensor:
        self.inp_dat = input_data
        return self.f(input_data)
    
    def backward(self, grad: Tensor) -> Tensor:
        return self.f_prime(self.inp_dat) * grad


def tanh(x:Tensor) -> Tensor:
    return np.tanh(x)


def tanh_prime(x:Tensor) -> Tensor:
    y = tanh(x)
    return 1 - y**2 


class Tanh(Activation):
    def __init__(self) -> None:
        super().__init__(tanh, tanh_prime)
