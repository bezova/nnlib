import numpy as np
from .tensor import Tensor

from typing import NamedTuple, Iterator

Batch = NamedTuple('Batch', [('input', Tensor),
                             ('target', Tensor)])

class DataIterator:
    def __call__(self, inputs:Tensor, targets:Tensor) -> Iterator[Batch]:
        raise NotImplementedError

class BatchIterator(DataIterator):
    def __init__(self, bs:int=32, shuffle:bool=True):
        self.bs = bs
        self.shuffle = shuffle
    
    def __call__(self, inputs:Tensor, targets:Tensor) -> Iterator[Batch]:
        starts = np.arange(0, len(inputs), self.bs)
        if self.shuffle: np.random.shuffle(starts)

        for start in starts:
            stop = start + self.bs
            batch_input = inputs[start:stop]
            batch_targ = targets[start:stop]
            yield Batch(input=batch_input, target=batch_targ)