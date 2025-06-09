from abc import abstractmethod
import numpy as np


class Optimizer:
    def __init__(self, init_lr, model) -> None:
        self.init_lr = init_lr
        self.model = model

    @abstractmethod
    def step(self):
        pass


class SGD(Optimizer):
    def __init__(self, init_lr, model):
        super().__init__(init_lr, model)
        
    
    def step(self):
        for layer in self.model.layers:
            if layer.optimizable:
                if layer.weight_decay and 'W' in layer.grads:
                    layer.grads['W'] += layer.weight_decay_lambda * layer.params['W']
                
                for key in layer.params.keys():
                    layer.params[key] -= self.init_lr * layer.grads[key]
                # 清空梯度
                layer.clear_grad()

class MomentGD(Optimizer):
    def __init__(self, init_lr, model, mu):
        """
        动量梯度下降优化器
        Args:
            init_lr (float): 初始学习率
            model (Layer): 模型实例
            mu (float): 动量系数 (0 < mu < 1)
        """
        super().__init__(init_lr, model)
        self.mu = mu  # 动量系数
        self.momentums = {}  # 存储各参数的速度变量
        
        # 为每个可优化层的参数初始化动量缓存
        for layer in self.model.layers:
            if layer.optimizable:
                self.momentums[layer] = {
                    'W': np.zeros_like(layer.params['W']),
                    'b': np.zeros_like(layer.params['b'])
                }

    def step(self):
        """
        执行一步参数更新
        """
        for layer in self.model.layers:
            if layer.optimizable:
                # 处理权重衰减（L2正则化）
                if layer.weight_decay and 'W' in layer.grads:
                    layer.grads['W'] += layer.weight_decay_lambda * layer.params['W']
                
                # 更新每个参数
                for key in ['W', 'b']:
                    grad = layer.grads[key]  # 当前梯度
                    # 动量更新公式: v_t = mu * v_{t-1} + grad
                    self.momentums[layer][key] = self.mu * self.momentums[layer][key] + grad
                    # 参数更新: param = param - lr * v_t
                    layer.params[key] -= self.init_lr * self.momentums[layer][key]
                
                # 清空当前梯度
                layer.clear_grad()