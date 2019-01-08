from .tensor import Tensor
from .layers import Layer
# import numpy as np
from typing import  Sequence, Iterator, Tuple

class NeuralNet(Layer):
    def __init__(self, layers: Sequence[Layer]) -> None:
        self.layers = layers

    def params_grads(self) -> Iterator[Tuple[Tensor, Tensor]]:
        for layer in self.layers:
            for name, param in layer.params.items():
                grad = layer.grads[name]
                yield param, grad

    def forward(self, data: Tensor) -> Tensor:
        for layer in self.layers:
            data = layer.forward(data)
        return data

    def backward(self, grad: Tensor) -> Tensor:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
            return grad