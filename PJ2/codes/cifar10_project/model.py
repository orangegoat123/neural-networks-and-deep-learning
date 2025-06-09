import torch.nn as nn
from configs import config
import torch

def get_activation():
    """根据配置返回激活函数实例"""
    if config.activation == 'leaky_relu':
        return nn.LeakyReLU(inplace=True)
    elif config.activation == 'elu':
        return nn.ELU(inplace=True)
    else:  # 默认ReLU
        return nn.ReLU(inplace=True)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        activation = get_activation()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels) if config.use_bn else nn.Identity()
        self.relu = activation
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels) if config.use_bn else nn.Identity()
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels) if config.use_bn else nn.Identity()
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class CustomCNN(nn.Module):
    def __init__(self):
        super().__init__()
        channels = config.customcnn_channels
        activation = get_activation()
        
        layers = []
        in_channels = 3
        for i, out_channels in enumerate(channels):
            layers += [
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels) if config.use_bn else nn.Identity(),
                activation
            ]
            # 只在每个通道后添加池化层（除了最后一个）
            if i < len(channels) - 1:
                layers.append(nn.MaxPool2d(2))
            in_channels = out_channels
        
        self.features = nn.Sequential(*layers)
        
        # 动态计算全连接层输入大小
        self.fc_input_size = self._get_fc_input_size()
        
        self.classifier = nn.Sequential(
            nn.Linear(self.fc_input_size, 512),
            activation,
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )
    
    def _get_fc_input_size(self):
        """动态计算全连接层输入大小"""
        with torch.no_grad():
            # 创建随机输入以获取特征图尺寸
            x = torch.randn(1, 3, 32, 32)
            output = self.features(x)
            return output.view(output.size(0), -1).size(1)
        
    def forward(self, x): 
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_channels = 64
        channels = config.resnet_channels
        activation = get_activation()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64) if config.use_bn else nn.Identity(),
            activation
        )
        
        # 创建多个残差层
        self.layer1 = self._make_layer(channels[0], 2, stride=1)
        self.layer2 = self._make_layer(channels[1], 2, stride=2)
        self.layer3 = self._make_layer(channels[2], 2, stride=2)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels[-1], 10)
    
    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def create_model():
    if config.model_type == 'resnet':
        return ResNet()
    else:
        return CustomCNN()

def load_model(model_path=config.model_path):
    """加载已训练模型"""
    model = create_model()
    model.load_state_dict(torch.load(model_path, map_location=config.device))
    model.to(config.device)
    model.eval()
    return model