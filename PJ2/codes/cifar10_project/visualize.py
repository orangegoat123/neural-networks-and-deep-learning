import matplotlib.pyplot as plt
from model import load_model
from configs import config

def visualize_filters():
    model = load_model()
    
    # 找到第一个卷积层
    if config.model_type == 'resnet':
        first_conv = model.conv1[0]  # ResNet的第一个卷积层在conv1中
    else:
        first_conv = model.features[0]  # CustomCNN的第一个卷积层
    
    filters = first_conv.weight.data.cpu()
    print("Filters shape:", filters.shape)  # 应为 [out_channels, in_channels, H, W]
    
    # 显示前16个滤波器的RGB通道
    n_filters = min(16, filters.shape[0])
    plt.figure(figsize=(10, 5))
    for i in range(n_filters):
        plt.subplot(4, 4, i+1)
        kernel = filters[i].permute(1, 2, 0)  # [H, W, C]
        kernel = (kernel - kernel.min()) / (kernel.max() - kernel.min())  # 归一化到[0,1]
        plt.imshow(kernel)
        plt.axis('off')
        plt.title(f'Filter {i+1}')
    plt.suptitle(f'First Conv Layer Filters ({config.model_type})', fontsize=16)
    plt.tight_layout()
    plt.savefig('conv_filters.png')
    plt.close()
    print("Filters visualization saved to conv_filters.png")