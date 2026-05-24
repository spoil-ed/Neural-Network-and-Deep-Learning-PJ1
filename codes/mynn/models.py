from .op import *
import pickle


def he_initializer(fan_in):
    def init(size):
        return np.random.normal(0.0, np.sqrt(2.0 / fan_in), size=size)
    return init


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

    def backward(self, loss_grad):
        return loss_grad.reshape(self.input_shape)

class Model_MLP(Layer):
    """
    A model with linear layers. We provied you with this example about a structure of a model.
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None):
        self.size_list = size_list
        self.act_func = act_func

        if size_list is not None and act_func is not None:
            self.layers = []
            for i in range(len(size_list) - 1):
                layer = Linear(
                    in_dim=size_list[i],
                    out_dim=size_list[i + 1],
                    initialize_method=he_initializer(size_list[i]),
                )
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                if act_func == 'Logistic':
                    raise NotImplementedError
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(size_list) - 2:
                    self.layers.append(layer_f)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]

        for i in range(len(self.size_list) - 1):
            self.layers = []
            for i in range(len(self.size_list) - 1):
                layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
                layer.W = param_list[i + 2]['W']
                layer.b = param_list[i + 2]['b']
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b
                layer.weight_decay = param_list[i + 2]['weight_decay']
                layer.weight_decay_lambda = param_list[i+2]['lambda']
                if self.act_func == 'Logistic':
                    raise NotImplemented
                elif self.act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(self.size_list) - 2:
                    self.layers.append(layer_f)
        
    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W' : layer.params['W'], 'b' : layer.params['b'], 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)
        

class Model_CNN(Layer):
    """
    A model with conv2D layers. Implement it using the operators you have written in op.py
    """
    def __init__(
        self,
        in_channels=1,
        input_size=28,
        num_classes=10,
        conv_channels=8,
        kernel_size=5,
        stride=1,
        padding=2,
        lambda_list=None,
    ):
        self.config = {
            'in_channels': in_channels,
            'input_size': input_size,
            'num_classes': num_classes,
            'conv_channels': conv_channels,
            'kernel_size': kernel_size,
            'stride': stride,
            'padding': padding,
        }
        conv = conv2D(
            in_channels=in_channels,
            out_channels=conv_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            initialize_method=he_initializer(in_channels * kernel_size * kernel_size),
        )
        out_size = (input_size + 2 * padding - kernel_size) // stride + 1
        linear_in = conv_channels * out_size * out_size
        linear = Linear(linear_in, num_classes, initialize_method=he_initializer(linear_in))
        if lambda_list is not None:
            conv.weight_decay = True
            conv.weight_decay_lambda = lambda_list[0]
            linear.weight_decay = True
            linear.weight_decay_lambda = lambda_list[1]
        self.layers = [conv, ReLU(), Flatten(), linear]

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        if X.ndim == 2:
            X = X.reshape(X.shape[0], self.config['in_channels'], self.config['input_size'], self.config['input_size'])
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads
    
    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            payload = pickle.load(f)

        if isinstance(payload, dict):
            self.__init__(**payload['config'])
            params = payload['params']
        else:
            raise ValueError("Unsupported CNN checkpoint format.")

        optimizable_layers = [layer for layer in self.layers if layer.optimizable]
        for layer, saved in zip(optimizable_layers, params):
            layer.W[...] = saved['W']
            layer.b[...] = saved['b']
            layer.weight_decay = saved.get('weight_decay', False)
            layer.weight_decay_lambda = saved.get('lambda', 1e-8)
        
    def save_model(self, save_path):
        params = []
        for layer in self.layers:
            if layer.optimizable:
                params.append({
                    'W': layer.params['W'],
                    'b': layer.params['b'],
                    'weight_decay': layer.weight_decay,
                    'lambda': layer.weight_decay_lambda,
                })
        with open(save_path, 'wb') as f:
            pickle.dump({'type': 'Model_CNN', 'config': self.config, 'params': params}, f)
