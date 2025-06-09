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
        self.W = np.random.randn(in_dim, out_dim) * np.sqrt(2 / in_dim) #  这里改为了适用relu的He初始化
        self.b = np.zeros((1, out_dim))
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
        batch_size = grad.shape[0]
        self.grads['W'] = (self.input.T @ grad)
        self.grads['b'] = np.sum(grad, axis=0)
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
        self.W = initialize_method(0, 0.1, (out_channels, in_channels, kernel_size, kernel_size))
        self.b = initialize_method(0, 0.1, (out_channels,))
        self.grads = {'W': None, 'b': None}
        self.input = None
        self.params = {'W': self.W, 'b': self.b}
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)
    
    def forward(self, X):
        """
        input X: [batch, channels, H, W]
        W : [out, in, k, k]
        """
        batch_size, in_chan, H_in, W_in = X.shape
        H_out = (H_in + 2*self.padding - self.kernel_size) // self.stride + 1
        W_out = (W_in + 2*self.padding - self.kernel_size) // self.stride + 1
        
        X_padded = np.pad(X, ((0,0), (0,0), (self.padding, self.padding), (self.padding, self.padding)))
        self.input = X_padded
        
        output = np.zeros((batch_size, self.out_channels, H_out, W_out))

        for h in range(H_out):
            h_start = h * self.stride
            h_end = h_start + self.kernel_size
            for w in range(W_out):
                w_start = w * self.stride
                w_end = w_start + self.kernel_size
                receptive_field = X_padded[:, :, h_start:h_end, w_start:w_end]
                output[:, :, h, w] = np.tensordot(receptive_field, self.W, axes=([1,2,3], [1,2,3])) + self.b
        
        return output

    def backward(self, grads):
        """
        grads : [batch_size, out_channel, new_H, new_W]
        """
        X_padded = self.input
        batch_size, _, H_pad, W_pad = X_padded.shape
        
        dW = np.zeros_like(self.W)
        db = np.sum(grads, axis=(0,2,3))
        
        for h in range(grads.shape[2]):
            h_start = h * self.stride
            for w in range(grads.shape[3]):
                w_start = w * self.stride
                receptive_field = X_padded[:, :, h_start:h_start+self.kernel_size, w_start:w_start+self.kernel_size]
                dW += np.tensordot(grads[:, :, h, w], receptive_field, axes=([0], [0]))
        
        self.grads['W'] = dW /batch_size
        self.grads['b'] = db
        
        dX_padded = np.zeros_like(X_padded)
        for h in range(grads.shape[2]):
            h_start = h * self.stride
            for w in range(grads.shape[3]):
                w_start = w * self.stride
                dX_padded[:, :, h_start:h_start+self.kernel_size, w_start:w_start+self.kernel_size] += \
                    np.tensordot(grads[:, :, h, w], self.W, axes=([1], [0]))
        
        if self.padding == 0:
            return dX_padded
        else:
            return dX_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
    
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
        return np.maximum(0, X)
    
    def backward(self, grads):
        return grads * (self.input > 0)

class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model = None, max_classes = 10) -> None:
        super().__init__()
        self.model = model
        self.max_classes = max_classes
        self.probs = None
        self.labels = None
        self.has_softmax = True
    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        exp_logits = np.exp(predicts - np.max(predicts, axis=1, keepdims=True))
        self.probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        self.labels = labels
        loss = -np.mean(np.log(self.probs[np.arange(len(labels)), labels] + 1e-8))
        return loss
    
    def backward(self):
        # first compute the grads from the loss to the input
        # / ---- your codes here ----/
        # Then send the grads to model for back propagation
        batch_size = self.probs.shape[0]
        grad = self.probs.copy()
        grad[np.arange(batch_size), self.labels] -= 1
        grad /= batch_size
        self.model.backward(grad)

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

class Flatten(Layer):
    def __init__(self):
        super().__init__()
        self.input_shape = None
        self.optimizable = False
    def __call__(self, X):
        return self.forward(X)
    def forward(self, X):
        self.input_shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, grad):
        return grad.reshape(self.input_shape)
    
    
class Dropout(Layer):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p  # 丢弃概率
        self.mask = None
        self.optimizable = False  # Dropout 层不需要优化参数

    def __call__(self, X, training=True):
        return self.forward(X, training)

    def forward(self, X, training=True):
        if training:
            self.mask = (np.random.rand(*X.shape) > self.p) / (1 - self.p)
            return X * self.mask
        else:
            return X

    def backward(self, grad):
        return grad * self.mask