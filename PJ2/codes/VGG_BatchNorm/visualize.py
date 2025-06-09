import numpy as np
import matplotlib.pyplot as plt
import os
import re
from pathlib import Path
import matplotlib as mpl

# 设置Agg后端 - 在导入pyplot之前设置
mpl.use('Agg')

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 设置路径
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # 假设visualize.py在codes/VGG_BatchNorm/目录下
MODELS_PATH = BASE_DIR / "reports" / "models"
FIGURES_PATH = BASE_DIR / "reports" / "figures"
FIGURES_PATH.mkdir(parents=True, exist_ok=True)  # 确保输出目录存在

# 学习率列表（根据您的文件名确定）
LEARNING_RATES = [0.001, 0.002, 0.0001, 0.0005]  # 1e-3, 2e-3, 1e-4, 5e-4

def parse_filename(filename):
    """从文件名解析模型类型和学习率"""
    match = re.match(r"(VGG_A|VGG_A_BatchNorm)_lr([\d.e-]+)_(losses|grads)\.txt", filename)
    if match:
        model_type = match.group(1)
        lr_str = match.group(2)
        
        # 处理科学计数法表示的学习率
        if 'e' in lr_str:
            lr = float(lr_str)
        else:
            # 处理没有小数点的学习率（如0.0001）
            lr = float(lr_str) if '.' in lr_str else float(lr_str) / 10000
            
        data_type = match.group(3)
        return model_type, lr, data_type
    
    return None, None, None

def load_all_data():
    """加载所有损失和梯度数据"""
    data = {
        "VGG_A": {"losses": {}, "grads": {}},
        "VGG_A_BatchNorm": {"losses": {}, "grads": {}}
    }
    
    for file_path in MODELS_PATH.glob("*.txt"):
        model_type, lr, data_type = parse_filename(file_path.name)
        if model_type and lr and data_type:
            # 确保学习率在列表中
            if lr not in LEARNING_RATES:
                # 查找最接近的学习率
                closest_lr = min(LEARNING_RATES, key=lambda x: abs(x - lr))
                print(f"警告: 学习率 {lr} 不在预设列表中，映射到最接近的 {closest_lr}")
                lr = closest_lr
                
            try:
                values = np.loadtxt(file_path)
                data[model_type][data_type][lr] = values
            except Exception as e:
                print(f"加载文件 {file_path} 时出错: {e}")
    
    return data

def plot_single_lr_comparison(data, lr=0.001):
    """图表1：单学习率下带BN和不带BN的训练对比"""
    plt.figure(figsize=(12, 10), dpi=100)
    
    # 上部分：损失曲线对比
    plt.subplot(2, 1, 1)
    plt.title(f"VGG_A 带BN与不带BN训练对比 (学习率={lr})")
    
    # 处理不带BN的数据
    if lr in data["VGG_A"]["losses"]:
        losses_without_bn = data["VGG_A"]["losses"][lr]
        # 使用滑动平均平滑曲线
        window_size = 100
        smoothed_without_bn = np.convolve(losses_without_bn, np.ones(window_size)/window_size, mode='valid')
        plt.plot(smoothed_without_bn, 'b-', label='无BN (平滑)')
        # 绘制前2000步的原始损失
        steps = min(2000, len(losses_without_bn))
        plt.plot(losses_without_bn[:steps], 'b:', alpha=0.5, label='无BN (原始)')
    
    # 处理带BN的数据
    if lr in data["VGG_A_BatchNorm"]["losses"]:
        losses_with_bn = data["VGG_A_BatchNorm"]["losses"][lr]
        window_size = 100
        smoothed_with_bn = np.convolve(losses_with_bn, np.ones(window_size)/window_size, mode='valid')
        plt.plot(smoothed_with_bn, 'r-', label='有BN (平滑)')
        steps = min(2000, len(losses_with_bn))
        plt.plot(losses_with_bn[:steps], 'r:', alpha=0.5, label='有BN (原始)')
    
    plt.xlabel('训练步数')
    plt.ylabel('损失值')
    plt.legend()
    plt.grid(True)
    
    # 下部分：梯度范数对比
    plt.subplot(2, 1, 2)
    
    # 处理不带BN的梯度数据
    if lr in data["VGG_A"]["grads"]:
        grads_without_bn = data["VGG_A"]["grads"][lr]
        window_size = 100
        smoothed_grads_without_bn = np.convolve(grads_without_bn, np.ones(window_size)/window_size, mode='valid')
        plt.plot(smoothed_grads_without_bn, 'b-', label='无BN梯度范数')
    
    # 处理带BN的梯度数据
    if lr in data["VGG_A_BatchNorm"]["grads"]:
        grads_with_bn = data["VGG_A_BatchNorm"]["grads"][lr]
        window_size = 100
        smoothed_grads_with_bn = np.convolve(grads_with_bn, np.ones(window_size)/window_size, mode='valid')
        plt.plot(smoothed_grads_with_bn, 'r-', label='有BN梯度范数')
    
    plt.xlabel('训练步数')
    plt.ylabel('梯度范数 (L2)')
    plt.yscale('log')  # 对数尺度更好展示差异
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    save_path = FIGURES_PATH / f'training_comparison_lr{lr}.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"训练对比图已保存到 {save_path}")

def plot_loss_landscape_comparison(data):
    """图表2：多学习率下带BN和不带BN的损失景观对比"""
    plt.figure(figsize=(12, 8), dpi=100)
    
    # 处理不带BN的模型
    if data["VGG_A"]["losses"]:
        without_bn_losses = []
        for lr in LEARNING_RATES:
            if lr in data["VGG_A"]["losses"]:
                without_bn_losses.append(data["VGG_A"]["losses"][lr])
        
        if without_bn_losses:
            # 确保长度一致
            min_length = min(len(l) for l in without_bn_losses)
            without_bn_losses = [l[:min_length] for l in without_bn_losses]
            
            # 计算最小最大曲线
            min_curve_without_bn = []
            max_curve_without_bn = []
            
            for i in range(min_length):
                step_losses = [l[i] for l in without_bn_losses]
                min_curve_without_bn.append(min(step_losses))
                max_curve_without_bn.append(max(step_losses))
            
            # 绘制不带BN的损失范围
            x = range(len(min_curve_without_bn))
            plt.fill_between(x, min_curve_without_bn, max_curve_without_bn, 
                            alpha=0.3, color='blue', label='无BN损失范围')
            plt.plot(x, min_curve_without_bn, 'b-', linewidth=1, alpha=0.7)
            plt.plot(x, max_curve_without_bn, 'b-', linewidth=1, alpha=0.7)
    
    # 处理带BN的模型
    if data["VGG_A_BatchNorm"]["losses"]:
        with_bn_losses = []
        for lr in LEARNING_RATES:
            if lr in data["VGG_A_BatchNorm"]["losses"]:
                with_bn_losses.append(data["VGG_A_BatchNorm"]["losses"][lr])
        
        if with_bn_losses:
            # 确保长度一致
            min_length = min(len(l) for l in with_bn_losses)
            with_bn_losses = [l[:min_length] for l in with_bn_losses]
            
            # 计算最小最大曲线
            min_curve_with_bn = []
            max_curve_with_bn = []
            
            for i in range(min_length):
                step_losses = [l[i] for l in with_bn_losses]
                min_curve_with_bn.append(min(step_losses))
                max_curve_with_bn.append(max(step_losses))
            
            # 绘制带BN的损失范围
            x = range(len(min_curve_with_bn))
            plt.fill_between(x, min_curve_with_bn, max_curve_with_bn, 
                            alpha=0.3, color='red', label='有BN损失范围')
            plt.plot(x, min_curve_with_bn, 'r-', linewidth=1, alpha=0.7)
            plt.plot(x, max_curve_with_bn, 'r-', linewidth=1, alpha=0.7)
    
    # 添加标签和标题
    plt.xlabel('训练步数')
    plt.ylabel('损失值')
    plt.title('多学习率下损失景观对比')
    plt.legend()
    plt.grid(True)
    
    save_path = FIGURES_PATH / 'loss_landscape_comparison.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"损失景观对比图已保存到 {save_path}")

def main():
    print("开始加载数据...")
    data = load_all_data()
    
    print("\n加载的数据摘要:")
    for model_type in data:
        print(f"模型: {model_type}")
        for data_type in data[model_type]:
            print(f"  {data_type.capitalize()}数据:")
            for lr, values in data[model_type][data_type].items():
                print(f"    学习率 {lr}: {len(values)}个数据点")
    
    # 生成单学习率对比图（使用第一个学习率）
    if LEARNING_RATES:
        lr = LEARNING_RATES[0]
        print(f"\n生成单学习率对比图 (学习率={lr})...")
        plot_single_lr_comparison(data, lr)
    
    # 生成损失景观对比图
    print("\n生成损失景观对比图...")
    plot_loss_landscape_comparison(data)
    
    print("\n分析完成！所有图表已保存到", FIGURES_PATH)

if __name__ == "__main__":
    main()