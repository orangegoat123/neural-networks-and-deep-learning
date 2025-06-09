import torch

class Config:
    # 数据路径
    data_root = './data'
    
    # 训练参数
    batch_size = 128
    epochs = 50
    lr = 0.001
    weight_decay = 1e-4
    
    # 模型配置
    model_type = 'resnet'  # 'customcnn' 或 'resnet'
    use_bn = True
    
    # 模型保存路径
    model_path = './checkpoints/best_model.pth'
    
    # 设备配置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 可视化配置
    plot_path = './plots/training_plot.png'
    
    # 不同神经元/过滤器数量
    customcnn_channels = [64, 128]  # CustomCNN各层通道数
    resnet_channels = [64, 128, 256]  # ResNet各层通道数
    
    # 不同损失函数
    loss_function = 'cross_entropy'  # 可选 'cross_entropy', 'mse', 'l1'
    l1_lambda = 0.001  # L1正则化系数
    
    # 不同激活函数
    activation = 'relu'  # 可选 'relu', 'leaky_relu', 'elu'
    
    # 不同优化器
    optimizer = 'adam'  # 可选 'sgd', 'adam', 'rmsprop'
    
config = Config()