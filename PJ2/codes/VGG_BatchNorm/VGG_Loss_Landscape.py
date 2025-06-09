import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
from torch import nn
import numpy as np
import torch
import os
import random
from tqdm import tqdm as tqdm
from IPython import display

from models.vgg import VGG_A
from models.vgg import VGG_A_BatchNorm # you need to implement this network
from data.loaders import get_cifar_loader

# ## Constants (parameters) initialization
num_workers = 4
batch_size = 128

# add our package dir to path 
module_path = os.path.dirname(os.getcwd())
home_path = module_path
figures_path = os.path.join(home_path, 'reports', 'figures')
models_path = os.path.join(home_path, 'reports', 'models')

device_id = 0
device = torch.device(f"cuda:{device_id}")




# This function is used to calculate the accuracy of model classification
def get_accuracy(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return 100.0 * correct / total

# Set a random seed to ensure reproducible results
def set_random_seeds(seed_value=0, device='cpu'):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu': 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# We use this function to complete the entire
# training process. In order to plot the loss landscape,
# you need to record the loss value of each step.
# Of course, as before, you can test your model
# after drawing a training round and save the curve
# to observe the training
def train(model, optimizer, criterion, train_loader, val_loader, scheduler=None, epochs_n=100, best_model_path=None):
    model.to(device)
    learning_curve = []  # 每轮的平均训练损失
    train_accuracy_curve = []  # 每轮的训练准确率
    val_accuracy_curve = []  # 每轮的验证准确率
    max_val_accuracy = 0  # 最高验证准确率
    max_val_accuracy_epoch = 0  # 达到最高验证准确率的轮次

    batches_n = len(train_loader)  # 每轮的批次数量
    losses_list = []  # 每个批次的损失值
    grads = []  # 每个批次的梯度范数
    
    for epoch in range(epochs_n):
        # 更新学习率（如果提供了调度器）
        if scheduler is not None:
            scheduler.step()
            
        model.train()  # 设置为训练模式
        epoch_loss = 0  # 当前轮次的总损失
        correct = 0  # 正确预测的数量
        total = 0  # 总样本数
        
        # 训练循环
        for i, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            prediction = model(X)
            loss = criterion(prediction, y)
            
            # 反向传播
            loss.backward()
            
            # 记录梯度和损失
            grad_norm = 0
            for param in model.parameters():
                if param.grad is not None:
                    # 计算梯度范数
                    grad_norm += torch.norm(param.grad).item()
            grads.append(grad_norm)
            losses_list.append(loss.item())
            
            # 更新权重
            optimizer.step()
            
            # 更新统计信息
            epoch_loss += loss.item()
            _, predicted = prediction.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
            
            # 每10个批次打印一次进度
            if i % 10 == 0:
                print(f'Epoch: {epoch+1}/{epochs_n}, Batch: {i}/{batches_n}, Loss: {loss.item():.4f}')
        
        # 计算训练准确率
        train_acc = 100.0 * correct / total
        train_accuracy_curve.append(train_acc)
        learning_curve.append(epoch_loss / batches_n)
        
        # 验证阶段
        model.eval()  # 设置为评估模式
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                loss = criterion(outputs, y)
                val_loss += loss.item()
                
                _, predicted = outputs.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()
        
        # 计算验证准确率
        val_acc = 100.0 * correct / total
        val_accuracy_curve.append(val_acc)
        
        # 保存最佳模型
        if val_acc > max_val_accuracy:
            max_val_accuracy = val_acc
            max_val_accuracy_epoch = epoch
            if best_model_path:
                torch.save(model.state_dict(), best_model_path)
                print(f"保存新的最佳模型，验证准确率: {val_acc:.2f}%")
        
        # 打印当前轮次统计信息
        print(f'Epoch {epoch+1}/{epochs_n}: '
              f'Train Loss: {epoch_loss/batches_n:.4f}, '
              f'Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {val_loss/len(val_loader):.4f}, '
              f'Val Acc: {val_acc:.2f}%')
    
    print(f"训练完成，最高验证准确率: {max_val_accuracy:.2f}% (第 {max_val_accuracy_epoch+1} 轮)")
    return learning_curve, train_accuracy_curve, val_accuracy_curve, losses_list, grads



# Use this function to plot the final loss landscape,
# fill the area between the two curves can use plt.fill_between()
def plot_loss_landscape(all_losses, labels):
    """
    绘制损失景观图，包括平均损失和最小-最大范围
    
    参数:
    all_losses: 包含多个损失列表的列表
    labels: 每个损失列表的标签
    """
    plt.figure(figsize=(12, 8))
    
    # 确保所有损失序列长度相同
    min_length = min(len(losses) for losses in all_losses)
    all_losses = [losses[:min_length] for losses in all_losses]
    
    for i, losses in enumerate(all_losses):
        # 计算最小、最大和平均曲线
        min_curve = np.min(losses, axis=0)
        max_curve = np.max(losses, axis=0)
        mean_curve = np.mean(losses, axis=0)
        x = np.arange(len(mean_curve))
        
        # 绘制平均曲线和最小-最大范围
        plt.plot(x, mean_curve, label=labels[i])
        plt.fill_between(x, min_curve, max_curve, alpha=0.2)
    
    plt.xlabel('训练步数')
    plt.ylabel('损失值')
    plt.title('损失景观比较')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(figures_path, 'loss_landscape_comparison.png'))
    plt.close()
    print("损失景观图已保存到 reports/figures/loss_landscape_comparison.png")
    
def main():
    
    # 确保目录存在 
    os.makedirs(figures_path, exist_ok=True)
    os.makedirs(models_path, exist_ok=True)
    # Initialize your data loader and
    # make sure that dataloader works
    # as expected by observing one
    # sample from it.
    train_loader = get_cifar_loader(train=True)   

    val_loader = get_cifar_loader(train=False)
    for X, y in train_loader:
        # 打印数据形状和范围
        print(f"输入数据形状: {X.shape}")
        print(f"标签数据形状: {y.shape}")
        print(f"输入数据范围: {X.min().item():.4f} 到 {X.max().item():.4f}")
        # 可视化一个样本
        plt.figure(figsize=(3, 3))
        plt.imshow(X[0].permute(1, 2, 0).numpy())
        plt.title(f"标签: {y[0].item()}")
        plt.savefig(os.path.join(figures_path, 'data_sample.png'))
        plt.close()
        print("样本图像已保存到 reports/figures/data_sample.png")
        break
    # 训练配置
    epo = 20
    learning_rates = [1e-3, 2e-3, 1e-4, 5e-4]
    models = {
        "VGG_A": VGG_A,
        "VGG_A_BatchNorm": VGG_A_BatchNorm
    }

    # 存储所有结果
    all_results = {}
    all_losses_for_landscape = []  # 用于绘制损失景观的所有损失值
    
    # 训练所有模型和所有学习率
    for model_name, model_class in models.items():
        model_results = []
        
        for lr in learning_rates:
            print(f"\n{'='*50}")
            print(f"训练模型: {model_name}, 学习率: {lr}")
            print(f"{'='*50}")
            
            # 设置随机种子
            set_random_seeds(seed_value=2020, device=device)
            
            # 初始化模型、优化器和损失函数
            model = model_class()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()
            
            # 训练模型
            results = train(
                model, optimizer, criterion,
                train_loader, val_loader,
                epochs_n=epo,
                best_model_path=os.path.join(models_path, f"{model_name}_lr{lr}.pth")
            )
            
            # 保存结果
            model_results.append(results)
            
            # 提取每个批次的损失值
            batch_losses = results[3]
            all_losses_for_landscape.append(batch_losses)
            
            # 保存损失和梯度到文件
            loss_save_path = os.path.join(models_path, f"{model_name}_lr{lr}_losses.txt")
            grad_save_path = os.path.join(models_path, f"{model_name}_lr{lr}_grads.txt")
            np.savetxt(loss_save_path, batch_losses, fmt='%s', delimiter=' ')
            np.savetxt(grad_save_path, results[4], fmt='%s', delimiter=' ')
            print(f"损失值已保存到: {loss_save_path}")
            print(f"梯度值已保存到: {grad_save_path}")
            
            # 计算并打印最终准确率
            model.load_state_dict(torch.load(os.path.join(models_path, f"{model_name}_lr{lr}.pth")))
            train_acc = get_accuracy(model, train_loader)
            val_acc = get_accuracy(model, val_loader)
            print(f"最终训练准确率: {train_acc:.2f}%")
            print(f"最终验证准确率: {val_acc:.2f}%")
        
        all_results[model_name] = model_results
    
    # 维护最小和最大曲线
    min_curve = []
    max_curve = []
    
    # 确保所有损失序列长度相同
    min_length = min(len(losses) for losses in all_losses_for_landscape)
    all_losses_for_landscape = [losses[:min_length] for losses in all_losses_for_landscape]
    
    # 计算每个步骤的最小和最大损失
    for step in range(min_length):
        step_losses = [losses[step] for losses in all_losses_for_landscape]
        min_curve.append(min(step_losses))
        max_curve.append(max(step_losses))
    
    # 保存最小和最大曲线
    np.savetxt(os.path.join(models_path, 'min_curve.txt'), min_curve, fmt='%s', delimiter=' ')
    np.savetxt(os.path.join(models_path, 'max_curve.txt'), max_curve, fmt='%s', delimiter=' ')
    print(f"最小曲线已保存到: {os.path.join(models_path, 'min_curve.txt')}")
    print(f"最大曲线已保存到: {os.path.join(models_path, 'max_curve.txt')}")
    
    # 可视化比较
    vgg_a_losses = [res[3] for res in all_results["VGG_A"]]
    vgg_bn_losses = [res[3] for res in all_results["VGG_A_BatchNorm"]]
    
    plot_loss_landscape(
        [vgg_a_losses, vgg_bn_losses],
        ["VGG_A (无BN)", "VGG_A_BatchNorm"]
    )
    
    # 绘制最小和最大曲线
    plt.figure(figsize=(12, 8))
    plt.plot(min_curve, label='最小损失')
    plt.plot(max_curve, label='最大损失')
    plt.fill_between(range(len(min_curve)), min_curve, max_curve, alpha=0.3, label='损失范围')
    plt.xlabel('训练步数')
    plt.ylabel('损失值')
    plt.title('损失景观最小-最大曲线')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(figures_path, 'min_max_curve.png'))
    plt.close()
    print("最小-最大曲线图已保存到 reports/figures/min_max_curve.png")

# 运行主函数
if __name__ == "__main__":
    main()