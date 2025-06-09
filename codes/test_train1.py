import mynn as nn
from draw_tools.plot import plot
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle

# 固定随机种子
np.random.seed(309)

# 数据加载与预处理
train_images_path = r'.\dataset\MNIST\train-images-idx3-ubyte.gz'
train_labels_path = r'.\dataset\MNIST\train-labels-idx1-ubyte.gz'

with gzip.open(train_images_path, 'rb') as f:
    magic, num, rows, cols = unpack('>4I', f.read(16))
    train_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28 * 28)
    
with gzip.open(train_labels_path, 'rb') as f:
    magic, num = unpack('>2I', f.read(8))
    train_labs = np.frombuffer(f.read(), dtype=np.uint8)

# 划分验证集
idx = np.random.permutation(np.arange(num))
train_imgs = train_imgs[idx]
train_labs = train_labs[idx]
valid_imgs = train_imgs[:10000]
valid_labs = train_labs[:10000]
train_imgs = train_imgs[10000:]
train_labs = train_labs[10000:]

# 数据归一化并调整为CNN输入形状 [batch, 1, 28, 28]
train_imgs = train_imgs.reshape(-1, 1, 28, 28) / 255.0
valid_imgs = valid_imgs.reshape(-1, 1, 28, 28) / 255.0

# 构建CNN模型
conv_layers = [
    {'in_channels': 1, 'out_channels': 16, 'kernel_size': 3, 'padding': 1, 'stride': 1},
    {'in_channels': 16, 'out_channels': 32, 'kernel_size': 3, 'padding': 1, 'stride': 2},
]
linear_layers = [
    {'in_dim': 32 * 14 * 14, 'out_dim': 128},  
    {'in_dim': 128, 'out_dim': 10}             
]
cnn_model = nn.models.Model_CNN(conv_layers=conv_layers, linear_layers=linear_layers)

# 配置优化器、学习率调度器和损失函数
optimizer = nn.optimizer.SGD(init_lr=0.01, model=cnn_model)
scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[800, 1600], gamma=0.5)
loss_fn = nn.op.MultiCrossEntropyLoss(model=cnn_model, max_classes=10)

# 训练运行器
runner = nn.runner.RunnerM(cnn_model, optimizer, nn.metric.accuracy, loss_fn, batch_size=64, scheduler=scheduler)

# 开始训练
runner.train([train_imgs, train_labs], [valid_imgs, valid_labs], 
             num_epochs=15, log_iters=100, save_dir='./best_cnn_model')

# 绘制训练曲线
_, axes = plt.subplots(1, 2)
axes.reshape(-1)
_.set_tight_layout(1)
plot(runner, axes)
plt.show()