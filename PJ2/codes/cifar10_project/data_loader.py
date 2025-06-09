import torch
from torchvision import datasets, transforms
from configs import config
from torch.utils.data import DataLoader

def get_loaders():
    # 训练集的数据增强（仅在训练时应用）
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),       # 随机裁剪
        transforms.RandomHorizontalFlip(p=0.5),      # 随机水平翻转
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 颜色扰动
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    
    # 验证集和测试集的预处理（无增强）
    transform_val_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    
    # 完整训练集（50,000张）
    full_train = datasets.CIFAR10(
        root=config.data_root, 
        train=True, 
        download=True, 
        transform=transform_train  # 注意：仅对训练集应用增强
    )
    
    # 划分训练集和验证集
    train_size = int(0.9 * len(full_train))
    val_size = len(full_train) - train_size
    train_set, val_set = torch.utils.data.random_split(full_train, [train_size, val_size])
    
    # 关键修复：验证集使用无增强的预处理
    # 因为 random_split 会继承原数据集的 transform，需手动覆盖
    val_set.dataset.transform = transform_val_test
    
    # 测试集（10,000张，无增强）
    test_set = datasets.CIFAR10(
        root=config.data_root, 
        train=False, 
        download=True, 
        transform=transform_val_test
    )
    
    # 创建 DataLoader
    train_loader = DataLoader(
        train_set, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=4
    )
    val_loader = DataLoader(
        val_set, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=4
    )
    test_loader = DataLoader(
        test_set, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=4
    )
    return train_loader, val_loader, test_loader