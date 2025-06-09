import mynn as nn
import numpy as np
from struct import unpack
import gzip
import pickle


conv_layers = [
    {'in_channels': 1, 'out_channels': 16, 'kernel_size': 3, 'padding': 1, 'stride': 1},
    {'in_channels': 16, 'out_channels': 32, 'kernel_size': 3, 'padding': 1, 'stride': 2},
]
linear_layers = [
    {'in_dim': 32 * 14 * 14, 'out_dim': 128},
    {'in_dim': 128, 'out_dim': 10}
]

# 初始化模型
model = nn.models.Model_CNN(conv_layers=conv_layers, linear_layers=linear_layers)

# 加载保存的参数
try:
    with open(r'./saved_models/best_cnn_model_final.pickle', 'rb') as f:
        saved_params = pickle.load(f)  # 加载参数列表
        
    model.load_model(saved_params)  # 调用你的加载方法
    print("模型加载成功！")
    
    # 验证参数加载情况
    print("\n参数校验：")
    param_count = 0
    for layer in model.layers:
        if isinstance(layer, nn.conv2D):
            print(f"卷积层{param_count} | 权重形状：{layer.W.shape}")
            param_count += 1
        elif isinstance(layer, nn.Linear):
            print(f"全连接层{param_count} | 权重形状：{layer.W.shape}")
            param_count += 1
except Exception as e:
    print(f"加载失败：{str(e)}")
    exit()

# 加载测试数据
test_images_path = r'./dataset/MNIST/t10k-images-idx3-ubyte.gz'
test_labels_path = r'./dataset/MNIST/t10k-labels-idx1-ubyte.gz'

with gzip.open(test_images_path, 'rb') as f:
    magic, num, rows, cols = unpack('>4I', f.read(16))
    test_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28 * 28)
    
with gzip.open(test_labels_path, 'rb') as f:
    magic, num = unpack('>2I', f.read(8))
    test_labs = np.frombuffer(f.read(), dtype=np.uint8)


test_imgs = test_imgs.reshape(-1, 1, 28, 28)  # 调整为 [N, C, H, W]
test_imgs = test_imgs.astype(np.float32) / 255.0  # 保持相同归一化


batch_size = 256
correct = 0
for i in range(0, len(test_imgs), batch_size):
    batch_x = test_imgs[i:i+batch_size]
    batch_y = test_labs[i:i+batch_size]
    
    # 前向传播
    logits = model(batch_x)
    
    # 计算准确率
    preds = np.argmax(logits, axis=1)
    correct += np.sum(preds == batch_y)

print(f"\n测试准确率：{correct/len(test_imgs):.2%}")