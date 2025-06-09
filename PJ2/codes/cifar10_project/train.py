import torch
from torch import optim
from tqdm import tqdm
from configs import config
from model import create_model
from data_loader import get_loaders
import torch.nn as nn
import matplotlib.pyplot as plt
import os

def train_model():
    # 确保绘图目录存在
    os.makedirs(os.path.dirname(config.plot_path), exist_ok=True)
    
    train_loader, val_loader, test_loader = get_loaders()
    model = create_model().to(config.device)
    
    # 根据配置选择优化器
    if config.optimizer == 'sgd':
        base_lr = config.lr * 10
        optimizer = optim.SGD(model.parameters(), lr=base_lr, 
                             momentum=0.9, weight_decay=config.weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[30, 45],  # 在第30和45个epoch降低学习率
        gamma=0.1  # 每次降低为原来的1/10
    )
    elif config.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=config.lr,
                                 weight_decay=config.weight_decay)
    else:  # 默认Adam
        optimizer = optim.Adam(model.parameters(), lr=config.lr, 
                              weight_decay=config.weight_decay)
    
    # 根据配置选择损失函数
    if config.loss_function == 'mse':
        criterion = nn.MSELoss()
    elif config.loss_function == 'l1':
        criterion = nn.L1Loss()
    else:  # 默认交叉熵
        criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    for epoch in range(config.epochs):
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        
        # 训练循环
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
            images, labels = images.to(config.device), labels.to(config.device)
            optimizer.zero_grad()
            outputs = model(images)
            
            # 计算损失
            if config.loss_function == 'cross_entropy':
                loss = criterion(outputs, labels)
            elif config.loss_function == 'mse':
                # 将标签转换为one-hot格式用于MSE
                target = torch.zeros_like(outputs)
                target.scatter_(1, labels.unsqueeze(1), 1.0)
                loss = criterion(outputs, target)
            else:  # L1损失
                loss = criterion(outputs, labels.float())
            
            # 添加L1正则化
            if config.l1_lambda > 0:
                l1_reg = torch.tensor(0., device=config.device)
                for param in model.parameters():
                    l1_reg += torch.norm(param, 1)
                loss += config.l1_lambda * l1_reg
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        # 验证集评估
        train_loss_avg = train_loss / len(train_loader)
        train_acc = correct / total
        val_acc = evaluate(model, val_loader)
        
        if scheduler:
               scheduler.step()
        # 记录历史数据
        history['train_loss'].append(train_loss_avg)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f'Epoch [{epoch+1}/{config.epochs}] | Loss: {train_loss_avg:.4f} '
              f'| Train Acc: {100*train_acc:.2f}% | Val Acc: {100*val_acc:.2f}%')
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), config.model_path)
    
    # 训练结束后绘制图表
    plot_training_history(history)

def evaluate(model, loader):
    """评估模型准确率"""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(config.device), labels.to(config.device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return correct / total

def plot_training_history(history):
    """绘制训练历史图表"""
    plt.figure(figsize=(12, 5))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Training & Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(config.plot_path)
    plt.close()
    print(f"Training plot saved to {config.plot_path}")