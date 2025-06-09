# visualize_cnn_weights.py
import mynn as nn
import numpy as np
import matplotlib.pyplot as plt
import pickle

# 配置区 ================================================
MODEL_PATH = r'./saved_models/best_cnn_model_final.pickle'  # 模型参数路径
CONV_LAYERS = [  # 必须与训练时的结构一致
    {'in_channels': 1, 'out_channels': 16, 'kernel_size': 3, 'padding': 1, 'stride': 1},
    {'in_channels': 16, 'out_channels': 32, 'kernel_size': 3, 'padding': 1, 'stride': 2},
]
LINEAR_LAYERS = [
    {'in_dim': 32 * 14 * 14, 'out_dim': 128},
    {'in_dim': 128, 'out_dim': 10}
]
# =======================================================

def load_cnn_model():
    """加载CNN模型结构并注入参数"""
    model = nn.models.Model_CNN(
        conv_layers=CONV_LAYERS,
        linear_layers=LINEAR_LAYERS
    )
    
    try:
        with open(MODEL_PATH, 'rb') as f:
            params = pickle.load(f)
            
        # 参数匹配到各层
        param_idx = 0
        for layer in model.layers:
            if isinstance(layer, (nn.conv2D, nn.Linear)):
                layer.W = params[param_idx]['W']
                layer.b = params[param_idx]['b']
                param_idx += 1
        print(f"成功加载{param_idx}个参数块")
        return model
    
    except Exception as e:
        print(f"加载失败: {str(e)}")
        exit()

def visualize_conv_kernels(layer, layer_idx):
    """可视化单个卷积层"""
    W = layer.W  # 形状 [out_channels, in_channels, k, k]
    out_ch, in_ch, k, _ = W.shape
    
    # 创建画布（修复坐标轴索引问题）
    fig, axes = plt.subplots(out_ch, in_ch, 
                           figsize=(1.5*in_ch, 1.5*out_ch),
                           squeeze=False)  # 保持二维结构
    axes = axes.reshape(out_ch, in_ch)  # 确保二维
    
    # 归一化颜色范围
    vmax = max(abs(W.min()), abs(W.max()))
    
    # 绘制每个核
    for oc in range(out_ch):
        for ic in range(in_ch):
            ax = axes[oc, ic]
            ax.imshow(W[oc, ic], 
                     cmap='coolwarm', 
                     vmin=-vmax, 
                     vmax=vmax,
                     interpolation='none')
            ax.set_xticks([])
            ax.set_yticks([])
            
            # 添加通道标签
            if oc == 0: ax.set_title(f'In {ic}', fontsize=6)
            if ic == 0: ax.set_ylabel(f'Out {oc}', fontsize=6, rotation=0, ha='right')
    
    plt.suptitle(f'Conv Layer {layer_idx} ({out_ch}×{in_ch}×{k}×{k})', y=0.95)
    plt.show()

def visualize_fc_weights(layer, layer_idx):
    """可视化全连接层"""
    W = layer.W
    
    # 尝试转换为图像格式
    if W.shape[0] == 32 * 14 * 14:  # 假设来自32通道14x14的特征图
        display_weights = W.reshape(32, 14, 14, -1)
        n_neurons = display_weights.shape[-1]
        
        plt.figure(figsize=(12, 8))
        for i in range(n_neurons):
            plt.subplot(8, 16, i+1)
            plt.imshow(display_weights[..., i].mean(axis=0), 
                      cmap='viridis', 
                      interpolation='nearest')
            plt.axis('off')
        plt.suptitle(f'FC Layer {layer_idx} Weight Patterns ({W.shape[0]}→{W.shape[1]})')
    else:
        plt.figure(figsize=(10, 6))
        plt.imshow(W, cmap='viridis', aspect='auto')
        plt.colorbar()
        plt.title(f'FC Layer {layer_idx} Weight Matrix ({W.shape[0]}×{W.shape[1]})')
    
    plt.tight_layout()
    plt.show()

def main():
    # 加载模型
    model = load_cnn_model()
    
    # 遍历各层进行可视化
    conv_idx = 0
    fc_idx = 0
    for i, layer in enumerate(model.layers):
        if isinstance(layer, nn.conv2D):
            print(f"可视化卷积层 {conv_idx}...")
            visualize_conv_kernels(layer, conv_idx)
            conv_idx += 1
        elif isinstance(layer, nn.Linear):
            print(f"可视化全连接层 {fc_idx}...")
            visualize_fc_weights(layer, fc_idx)
            fc_idx += 1

if __name__ == '__main__':
    main()