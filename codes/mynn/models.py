from .op import *
import pickle

class Model_MLP(Layer):
    """
    A model with linear layers. We provied you with this example about a structure of a model.
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None,dropout_rates=None):
        self.size_list = size_list
        self.act_func = act_func
        self.dropout_rates = dropout_rates

        if size_list is not None and act_func is not None:
            self.layers = []
            for i in range(len(size_list) - 1):
                layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1])
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
                    if dropout_rates is not None and i < len(dropout_rates):
                        dropout = Dropout(p=dropout_rates[i])
                        self.layers.append(dropout)

    def __call__(self, X, training=True):
        return self.forward(X, training=training)

    def forward(self, X,training=True):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            if isinstance(layer, Dropout):
                outputs = layer(outputs, training=training)
            else:
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
        self.dropout_rates = param_list[2]
        self.layers = []
        param_idx = 3
        for i in range(len(self.size_list) - 1):
            layer = Linear(self.size_list[i], self.size_list[i+1])
            current_params = param_list[param_idx]  # 关键修改点
            layer.W = current_params['W']
            layer.b = current_params['b']
            layer.weight_decay = current_params['weight_decay']
            layer.weight_decay_lambda = current_params['lambda']
            self.layers.append(layer)
            param_idx += 1  # 索引递增
         # 添加激活层和Dropout层
            if i < len(self.size_list)-2:
                if self.act_func == 'ReLU':
                    self.layers.append(ReLU())
                #
                elif self.act_func == 'Logistic':
                    self.layers.append(Sigmoid())
                
                # 添加Dropout
                if self.dropout_rates and i < len(self.dropout_rates):
                    self.layers.append(Dropout(p=self.dropout_rates[i]))
        
    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func,self.dropout_rates]
        for layer in self.layers:
            if isinstance(layer, Linear) and hasattr(layer, 'params'):
                param_list.append({
                'W': layer.params['W'],
                'b': layer.params['b'],
                'weight_decay': layer.weight_decay,
                'lambda': layer.weight_decay_lambda
            })
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)
        


class Model_CNN(Layer):
    def __init__(self, conv_layers=None, linear_layers=None):
        super().__init__()
        self.layers = []
        
        # 仿照MLP的构造方式
        if conv_layers:
            for config in conv_layers:
                self.layers.append(conv2D(**config))
                self.layers.append(ReLU())
        
        self.layers.append(Flatten())
        
        if linear_layers:
            for i in range(len(linear_layers)-1):
                self.layers.append(Linear(**linear_layers[i]))
                self.layers.append(ReLU())
            self.layers.append(Linear(**linear_layers[-1]))

    def __call__(self, X):
        return self.forward(X)
    def forward(self, X):
        for layer in self.layers:
            X = layer(X)
        return X

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def load_model(self, param_list):
        # 类似MLP的参数加载逻辑
        idx = 0
        for layer in self.layers:
            if isinstance(layer, (conv2D, Linear)):
                layer.W = param_list[idx]['W']
                layer.b = param_list[idx]['b']
                idx += 1

