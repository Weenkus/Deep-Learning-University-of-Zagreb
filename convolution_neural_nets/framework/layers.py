from abc import ABCMeta, abstractmethod
import numpy as np
import scipy
import scipy.stats as stats

from im2col_cython import col2im_cython, im2col_cython

zero_init = np.zeros


def variance_scaling_initializer(shape, fan_in, factor=2.0, seed=None):
    sigma = np.sqrt(factor / fan_in)
    # x = stats.truncnorm(-max_val*sigma, max_val*sigma, loc=0, scale=sigma)
    return stats.truncnorm(-2, 2, loc=0, scale=sigma).rvs(shape)


# -- ABSTRACT CLASS DEFINITION --
class Layer(metaclass=ABCMeta):
    "Interface for layers"

    # See documentation of abstract base classes (ABC): https://docs.python.org/3/library/abc.html

    @abstractmethod
    def forward(self, inputs):
        """
        Args:
          inputs: ndarray tensor.
        Returns:
          ndarray tensor, result of the forward pass.
        """
        pass

    @abstractmethod
    def backward_inputs(self, grads):
        """
        Args:
          grads: gradient of the loss with respect to the output of the layer.
        Returns:
          Gradient of the loss with respect to the input of the layer.
        """
        pass

    def backward_params(self, grads):
        """
        Args:
          grads: gradient of the loss with respect to the output of the layer.
        Returns:
          Gradient of the loss with respect to all the parameters of the layer as a list
          [[w0, g0], ..., [wk, gk], self.name] where w are parameter weights and g their gradient.
        """
        pass


# -- CONVOLUTION LAYER --
class Convolution(Layer):
    "N-dimensional convolution layer"

    def __init__(self, input_layer, num_filters, kernel_size, name, padding='SAME',
                 weights_initializer_fn=variance_scaling_initializer,
                 bias_initializer_fn=zero_init):
        # shape of the output is [num_filters, height, width]

        self.input_shape = input_layer.shape
        N, C, H, W = input_layer.shape
        self.C = C
        self.N = N
        self.num_filters = num_filters
        self.kernel_size = kernel_size

        assert kernel_size % 2 == 1

        self.padding = padding
        if padding == 'SAME':
            self.shape = (N, num_filters, H, W)
        else:
            self.shape = (N, num_filters, H - kernel_size + 1, W - kernel_size + 1)

        fan_in = C * kernel_size ** 2
        self.weights = weights_initializer_fn([num_filters, kernel_size ** 2 * C], fan_in)
        # print(self.weights.shape)
        self.bias = bias_initializer_fn([num_filters])
        self.im2col_data = np.zeros([kernel_size ** 2 * C, N * H * W])
        self.im2col_data_backward = np.zeros([kernel_size ** 2 * self.num_filters, N * H * W])
        self.pad = (kernel_size - 1) // 2
        # this implementation doesn't support strided convolutions
        self.stride = 1
        self.name = name
        self.has_params = True

    def forward(self, x):
        k = self.kernel_size
        self.x_cols = im2col_cython(x, k, k, self.pad, self.stride)
        res = self.weights.dot(self.x_cols) + self.bias.reshape(-1, 1)
        N, C, H, W = x.shape
        out = res.reshape(self.num_filters, H, W, N)
        return out.transpose(3, 0, 1, 2)

    def backward_inputs(self, grad_out):
        pad = self.pad
        # nice trick from CS231n, backward pass can be done with just matrix mul and col2im
        grad_out = grad_out.transpose(1, 2, 3, 0).reshape(self.num_filters, -1)
        grad_x_cols = self.weights.T.dot(grad_out)
        N, C, H, W = self.input_shape
        k = self.kernel_size
        grad_x = col2im_cython(grad_x_cols, N, C, H, W, k, k, pad, self.stride)
        return grad_x
        # return grad_x

    def backward_params(self, grad_out):
        grad_bias = np.sum(grad_out, axis=(0, 2, 3))
        grad_out = grad_out.transpose(1, 2, 3, 0).reshape(self.num_filters, -1)
        # grad_weights = grad_out.dot(self.x_cols.T).reshape(self.weights.shape)
        grad_weights = grad_out.dot(self.x_cols.T).reshape(self.weights.shape)
        # weight_decay = 1e-1
        # grad_weights += weight_decay * self.weights
        return [[self.weights, grad_weights], [self.bias, grad_bias], self.name]


class MaxPooling(Layer):
    def __init__(self, input_layer, name, pool_size=2, stride=2):
        # self.num_inputs = self.input_shape[0]
        self.name = name
        self.input_shape = input_layer.shape
        N, C, H, W = self.input_shape
        self.stride = stride
        self.shape = (N, C, H // stride, W // stride)
        self.pool_size = pool_size
        assert pool_size == stride, 'Invalid pooling params'
        assert H % pool_size == 0
        assert W % pool_size == 0
        self.im2col_data = np.zeros([N * H * W, pool_size ** 2 * C])
        self.name = name
        self.im2col_grads = np.array([1])
        self.grad_inputs = np.zeros(self.input_shape)
        self.has_params = False

    def forward(self, x):
        N, C, H, W = x.shape
        self.input_shape = x.shape
        self.x = x.reshape(N, C, H // self.pool_size, self.pool_size,
                           W // self.pool_size, self.pool_size)
        self.out = self.x.max(axis=3).max(axis=4)
        return self.out.copy()

    def backward_inputs(self, grad_out):
        grad_x = np.zeros_like(self.x)
        out_newaxis = self.out[:, :, :, np.newaxis, :, np.newaxis]
        mask = (self.x == out_newaxis)
        dout_newaxis = grad_out[:, :, :, np.newaxis, :, np.newaxis]
        dout_broadcast, _ = np.broadcast_arrays(dout_newaxis, grad_x)
        grad_x[mask] = dout_broadcast[mask]
        #print(np.sum(mask, axis=(3, 5), keepdims=True))
        #print(np.sum(mask, axis=(3, 5), keepdims=True).shape)
        grad_x /= np.sum(mask, axis=(3, 5), keepdims=True)
        grad_x = grad_x.reshape(self.input_shape)
        return grad_x


class Flatten(Layer):
    def __init__(self, input_layer, name):
        self.input_shape = input_layer.shape
        self.N = self.input_shape[0]
        self.num_outputs = 1
        for i in range(1, len(self.input_shape)):
            self.num_outputs *= self.input_shape[i]
        self.shape = (self.N, self.num_outputs)
        self.has_params = False
        self.name = name

    def forward(self, inputs):
        self.input_shape = inputs.shape
        inputs_flat = inputs.reshape(self.input_shape[0], -1)
        self.shape = inputs_flat.shape
        return inputs_flat

    def backward_inputs(self, grads):
        return grads.reshape(self.input_shape)


class FC(Layer):
    def __init__(self, input_layer, num_outputs, name,
                 weights_initializer_fn=variance_scaling_initializer,
                 bias_initializer_fn=zero_init):
        """
        Args:
          input_layer
        """

        self.input_shape = input_layer.shape
        self.N = self.input_shape[0]
        self.shape = (self.N, num_outputs)
        self.num_outputs = num_outputs

        self.num_inputs = 1
        for i in range(1, len(self.input_shape)):
            self.num_inputs *= self.input_shape[i]

        self.weights = weights_initializer_fn([num_outputs, self.num_inputs], fan_in=self.num_inputs)
        self.bias = bias_initializer_fn([num_outputs])
        self.name = name
        self.has_params = True
        self.inputs = None

    def forward(self, inputs):
        """
        Args:
          inputs: ndarray of shape (N, num_inputs)
        Returns:
          An ndarray of shape (N, num_outputs)
        """
        self.inputs = inputs
        output = np.dot(inputs, self.weights.T) + self.bias
        return output

    def backward_inputs(self, grads):
        """
        Args:
          grads: ndarray of shape (N, num_outputs)
        Returns:
          An ndarray of shape (N, num_inputs)
        """
        return np.dot(grads, self.weights)

    def backward_params(self, grads):
        """
        Args:
          grads: ndarray of shape (N, num_outputs)
        Returns:
          List of params and gradient pairs.
        """
        grad_weights = np.dot(grads.T, self.inputs)
        grad_bias = np.sum(grads, axis=0)
        return [[self.weights, grad_weights], [self.bias, grad_bias], self.name]


class ReLU(Layer):
    def __init__(self, input_layer, name):
        self.shape = input_layer.shape
        self.name = name
        self.has_params = False
        self.mask = None

    def forward(self, inputs):
        """
        Args:
          inputs: ndarray of shape (N, C, H, W).
        Returns:
          ndarray of shape (N, C, H, W).
        """
        out = np.maximum(0, inputs)
        self.mask = inputs
        return out

    def backward_inputs(self, grads):
        """
        Args:
          grads: ndarray of shape (N, C, H, W).
        Returns:
          ndarray of shape (N, C, H, W).
        """
        return grads * (self.mask >= 0)


class SoftmaxCrossEntropyWithLogits():
    def __init__(self):
        self.has_params = False
        self.probs = None

    def forward(self, x, y):
        """
        Args:
          x: ndarray of shape (N, num_classes).
          y: ndarray of shape (N, num_classes).
        Returns:
          Scalar, average loss over N examples.
          It is better to compute average loss here instead of just sum
          because then learning rate and weight decay won't depend on batch size.

        """
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.probs = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        N = x.shape[0]

        loss = - np.average(np.log(self.probs[range(N), np.where(y == 1)[1]]))
        return loss

    def backward_inputs(self, x, y):
        """
        Args:
          x: ndarray of shape (N, num_classes).
          y: ndarray of shape (N, num_classes).
        Returns:
          Gradient with respect to the x, ndarray of shape (N, num_classes).
        """
        # Hint: don't forget that we took the average in the forward pass
        N = x.shape[0]
        probs = self.probs.copy()
        probs[range(N), np.where(y == 1)[1]] -= 1
        return probs / N


class L2Regularizer():
    def __init__(self, weights, weight_decay, name):
        """
        Args:
          weights: parameters which will be regularizerized
          weight_decay: lambda, regularization strength
          name: layer name
        """
        # this is still a reference to original tensor so don't change self.weights
        self.weights = weights
        self.weight_decay = weight_decay
        self.name = name

    def forward(self):
        """
         Returns:
          Scalar, loss due to the L2 regularization.
        """
        return self.weights * self.weight_decay

    def backward_params(self):
        """
        Returns:
          Gradient of the L2 loss with respect to the regularized weights.
        """
        grad_weights = np.sum(self.weights)
        return [[self.weights, grad_weights], self.name]


class RegularizedLoss(object):
    def __init__(self, data_loss, regularizer_losses):
        self.data_loss = data_loss
        self.regularizer_losses = regularizer_losses
        self.has_params = True
        self.name = 'RegularizedLoss'

    def forward(self, x, y):
        loss_val = self.data_loss.forward(x, y)
        for loss in self.regularizer_losses:
            loss_val += loss.forward()
        return loss_val

    def backward_inputs(self, x, y):
        return self.data_loss.backward_inputs(x, y)

    def backward_params(self):
        grads = []
        for loss in self.regularizer_losses:
            grads += [loss.backward_params()]
        return grads
