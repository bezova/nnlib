from .tensor import Tensor

import numpy as np

class Loss():
    def loss(self, predicted: Tensor, actual: Tensor) -> Tensor:
        ''' calculate loss function
        '''
        raise NotImplementedError

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        ''' calculate gradient of loss function in regard to predicted
        '''
        raise NotImplementedError

class MSE(Loss):
    ''' mean squared error
    '''
    def loss(self, predicted: Tensor, actual: Tensor) -> Tensor:
        ''' calculate MSE loss function
        '''
        return np.sum( (predicted-actual)**2 )

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        ''' calculate gradient of loss function in regard to predicted
        '''
        return 2*(predicted - actual)
