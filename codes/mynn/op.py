from abc import abstractmethod
import numpy as np

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
    
    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.W = initialize_method(size=(in_dim, out_dim))
        self.b = initialize_method(size=(1, out_dim))
        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.

        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
            
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        self.input = X
        return X @ self.W + self.b

    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        assert self.input is not None, "forward must be called before backward."
        self.grads['W'] = self.input.T @ grad
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True)
        return grad @ self.W.T
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

class conv2D(Layer):
    """
    The 2D convolutional layer. Try to implement it on your own.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.W = initialize_method(size=(out_channels, in_channels, kernel_size, kernel_size))
        self.b = initialize_method(size=(out_channels,))
        self.grads = {'W' : None, 'b' : None}
        self.params = {'W' : self.W, 'b' : self.b}
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda
        self.input = None
        self.input_padded = None
        self.cols = None
        self.out_h = None
        self.out_w = None

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)
    
    def forward(self, X):
        """
        input X: [batch, channels, H, W]
        W : [1, out, in, k, k]
        no padding
        """
        assert X.ndim == 4, "conv2D expects input [batch, channels, H, W]."
        assert X.shape[1] == self.in_channels
        self.input = X

        if self.padding > 0:
            X_pad = np.pad(
                X,
                ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                mode='constant',
            )
        else:
            X_pad = X
        self.input_padded = X_pad

        batch, channels, height, width = X_pad.shape
        self.out_h = (height - self.kernel_size) // self.stride + 1
        self.out_w = (width - self.kernel_size) // self.stride + 1

        shape = (batch, channels, self.out_h, self.out_w, self.kernel_size, self.kernel_size)
        strides = (
            X_pad.strides[0],
            X_pad.strides[1],
            self.stride * X_pad.strides[2],
            self.stride * X_pad.strides[3],
            X_pad.strides[2],
            X_pad.strides[3],
        )
        windows = np.lib.stride_tricks.as_strided(X_pad, shape=shape, strides=strides)
        self.cols = windows.transpose(0, 2, 3, 1, 4, 5).reshape(
            batch * self.out_h * self.out_w,
            channels * self.kernel_size * self.kernel_size,
        )
        output = self.cols @ self.W.reshape(self.out_channels, -1).T + self.b
        return output.reshape(batch, self.out_h, self.out_w, self.out_channels).transpose(0, 3, 1, 2)

    def backward(self, grads):
        """
        grads : [batch_size, out_channel, new_H, new_W]
        """
        assert self.input is not None and self.input_padded is not None, "forward must be called before backward."

        batch, _, out_h, out_w = grads.shape
        grads_col = grads.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)
        dW = grads_col.T @ self.cols
        dW = dW.reshape(self.W.shape)
        db = np.sum(grads, axis=(0, 2, 3))
        dcols = grads_col @ self.W.reshape(self.out_channels, -1)
        dcols = dcols.reshape(
            batch,
            out_h,
            out_w,
            self.in_channels,
            self.kernel_size,
            self.kernel_size,
        )
        dX_pad = np.zeros_like(self.input_padded)
        for i in range(out_h):
            h_start = i * self.stride
            h_end = h_start + self.kernel_size
            for j in range(out_w):
                w_start = j * self.stride
                w_end = w_start + self.kernel_size
                dX_pad[:, :, h_start:h_end, w_start:w_end] += dcols[:, i, j]

        self.grads['W'] = dW
        self.grads['b'] = db

        if self.padding > 0:
            return dX_pad[:, :, self.padding:-self.padding, self.padding:-self.padding]
        return dX_pad
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}
        
class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output

class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model = None, max_classes = 10) -> None:
        super().__init__()
        self.model = model
        self.max_classes = max_classes
        self.has_softmax = True
        self.predicts = None
        self.labels = None
        self.probs = None
        self.grads = None
        self.optimizable = False

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        self.predicts = predicts
        self.labels = labels.astype(int)
        if self.has_softmax:
            self.probs = softmax(predicts)
        else:
            self.probs = predicts
        batch_indices = np.arange(predicts.shape[0])
        eps = 1e-12
        return -np.mean(np.log(self.probs[batch_indices, self.labels] + eps))
    
    def backward(self):
        # first compute the grads from the loss to the input
        batch_size = self.predicts.shape[0]
        labels_one_hot = np.zeros((batch_size, self.max_classes))
        labels_one_hot[np.arange(batch_size), self.labels] = 1
        if self.has_softmax:
            self.grads = (self.probs - labels_one_hot) / batch_size
        else:
            self.grads = -labels_one_hot / (self.probs + 1e-12) / batch_size
        # Then send the grads to model for back propagation
        if self.model is not None:
            self.model.backward(self.grads)
        return self.grads

    def cancel_soft_max(self):
        self.has_softmax = False
        return self
    
class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    pass
       
def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition
