from .tensor import Tensor
from .nn import NeuralNet
from .data import DataIterator, BatchIterator
from loss import Loss, MSE


def train_nn(
    net: NeuralNet, 
    epochs: int = 1, 
    inputs: Tensor, 
    targets: Tensor,
    loss: Loss = MSE(),
    batch_iter: DataIterator = BatchIterator(),
    optimizer: Optimizer = SGD(),
    ) -> None:

    for epoch in range(epochs):
        epoch_loss = 0.

        for batch in batch_iter(inputs, targets):

            pred = net.forward(batch.input)

            batch_loss = loss.loss(pred, batch.target)
            epoch_loss += batch_loss

            loss_grad = loss.grad(pred, batch.target)

            net_grad = net.backward(loss_grad)

            optimizer.step(net)
        
        print(f'epoch:{epoch}, loss:{eoch_loss}')


