"""CIFAR-10 with ResNet-18 图像分类消融实验"""

import os
import time
import warnings
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from loguru import logger

from .early_stop import EarlyStopping


# 禁用警告
warnings.filterwarnings(
    "ignore", message="enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True"
)


# ==================== 主配置字典 ====================
config = {
    # 优化器开关
    'optimizer_type': 'MetaAdamW',  # 'AdamW', 'MetaAdamW'

    # 实验设置
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'seed': 42,

    # ResNet-18 图像分类特定配置
    'batch_size': 128,
    'dropout': 0.2,

    # 优化器通用设置
    'lr': 5e-4,
    'weight_decay': 0.01,

    # MetaAdamW 专属设置
    'meta_update_freq': 391,            # 元更新频率（步数），391 steps/epoch
    'feature_dim': 6,                   # 每个参数组的特征维度（在 'enhanced' 时需设置 16 以确保足够容纳）；若为 None 则自动计算有效特征维度
    'attn_hidden_dim': 64,              # 注意力MLP隐藏维度 16
    'attn_layers': 64,                   # 注意力层数 8
    'attn_heads': 6,                    # 注意力头数
    'meta_lr': 1e-4,                    # 元学习的学习率
    'group_strategy': 'fine_grained',   # 是否启用精细分组：'original' 或 'fine_grained'
    # basic 核心：梯度范数、动量范数、参数范数、梯度与动量的余弦相似度
    # 特征表达： 'basic', 'norm_basic', 'basic_plus', 'norm_basic_plus', 'enhanced'
    'feature_version': 'basic',         # basic: 4 or+1 or+1, basic_plus: 4+4 or+1 or+2, enhanced: 9 or+1 + group_embed_dim
    'use_v_norms': True,                # basic 相关特征中是否启用 二阶动量范数 特征
    'use_feature_gating': False,        # 是否启用 特征门控 机制
    'feature_gating_sparsity': 1e-4,    # 启用特征门控时的正则化强度
    'include_time_step': True,          # 特征中是否包含时间步
    'group_embed_dim': 4,               # 组嵌入向量的维度（仅在 'enhanced' 时使用）
    # 分别对应：梯度方向（使元学习更新的方向更优），损失下降（让模型泛化更好），泛化差距（减小过拟合），三者混合
    'meta_objective': 'combined',       # 元学习目标：'gradient', 'loss_decrease', 'gen_gap', 'combined'

    # 元学习目标为 'combined' 时：辅助损失组合方案
    'aux_loss_fixed_weights': [1.0, 0.3, 0.3],   # 固定权重，若为 None 则使用简单相加
    'use_huw': True,                             # 是否启用同调不确定性加权（优先级高于固定权重）
    'task_types': ['regression', 'regression', 'regression'],
    'huw_priorities': [5.0, 1.0, 2.0],           # HUW 业务优先级，默认等权重

    # 训练设置
    'epochs': 15,
    'log_interval': 10,
    # 一般3轮时主模型大致可以收敛到较平坦区域，然后再启用注意力模块，这样可以避免早期的不稳定影响后续元学习
    'warmup_epoch': 1,                           # 在主模型训练到 第几轮 时，优化器才启动元学习

    # 早停设置
    'patience': 2,
    'min_delta': 1e-3,
    'restore_best_weights': True,
}


# 定义适用于 CV 任务的 loss_fn 闭包
# 统一接收 (model, batch)，在内部解包
criterion = nn.CrossEntropyLoss()
def cv_loss_fn(eval_model, batch):
    x, y = batch
    outputs = eval_model(x)
    return criterion(outputs, y)


# ==================== ResNet-18 模型（适配 CIFAR-10） ====================
class ResNet18(nn.Module):
    """
    基于 torchvision ResNet-18 修改，适配 CIFAR-10 (32x32 输入)
    CNN 类型
    """
    def __init__(self, num_classes=10, dropout=0.0):
        super().__init__()
        # 从头训练
        self.model = torchvision.models.resnet18(pretrained=False)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.model(x)
        x = self.dropout(x)
        return x


# ==================== 数据集准备 ====================
def get_dataloaders(batch_size, num_workers=2):
    """
    加载 CIFAR-10 训练集和测试集（作为验证集）
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(testset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    logger.info(f"训练样本数: {len(trainset)}, 验证样本数: {len(testset)}")
    return train_loader, val_loader


# ==================== 训练函数 ====================
def train_epoch(model, device, train_loader, optimizer, epoch,
                hyperadamw=False, meta_update_freq=None):
    """
    训练一个 epoch，返回 (平均损失, 平均准确率, 平均辅助损失)
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    aux_loss_total = 0.0
    aux_loss_count = 0
    aux_loss = 0.0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch}', leave=True, ncols=140)

    if hyperadamw and meta_update_freq is not None and meta_update_freq > 0 and epoch >= config.get('warmup_epoch', 1):
        # 创建两个迭代器用于元学习
        train_iter1 = iter(train_loader)
        train_iter2 = iter(train_loader)

    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # 统计
        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        _, predicted = outputs.max(1)
        total_correct += predicted.eq(targets).sum().item()
        total_samples += batch_size

        # 元学习更新
        if hyperadamw and meta_update_freq and meta_update_freq > 0 and epoch >= config.get('warmup_epoch', 1):
            if (batch_idx + 1) % meta_update_freq == 0:
                try:
                    data1, target1 = next(train_iter1)
                except StopIteration:
                    train_iter1 = iter(train_loader)
                    data1, target1 = next(train_iter1)
                try:
                    data2, target2 = next(train_iter2)
                except StopIteration:
                    train_iter2 = iter(train_loader)
                    data2, target2 = next(train_iter2)

                data1, target1 = data1.to(device), target1.to(device)
                data2, target2 = data2.to(device), target2.to(device)

                aux_loss = optimizer.update_attention(
                    model,
                    (data1, target1),
                    (data2, target2),
                    cv_loss_fn
                )
                aux_loss_total += aux_loss
                aux_loss_count += 1

        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.0 * total_correct / total_samples:.2f}%',
            'aux_loss': f'{aux_loss:.4f}'
        })

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    avg_aux_loss = aux_loss_total / aux_loss_count if aux_loss_count > 0 else 0.0
    return avg_loss, avg_acc, avg_aux_loss


def validate(model, device, val_loader):
    """
    验证，返回 (平均损失, 准确率)
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            _, predicted = outputs.max(1)
            total_correct += predicted.eq(targets).sum().item()
            total_samples += batch_size

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


# ==================== 主程序 ====================
def train_cifar10():
    # 配置优化器
    optimizer_type = config['optimizer_type']

    # 日志
    device = torch.device(config['device'])
    logger.info(f"Device: {device}")
    logger.info(f"Optimizer: {optimizer_type}")

    # 加载数据
    train_loader, val_loader = get_dataloaders(
        batch_size=config['batch_size'],
        num_workers=config.get('num_workers', 2)
    )

    # 创建模型
    model = ResNet18(num_classes=10, dropout=config.get('dropout', 0.0)).to(device)

    # 优化器
    if optimizer_type == 'AdamW':
        from torch.optim import AdamW
        optimizer = AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        hyperadamw = False

    else:
        from .meta_adamw import MetaAdamW
        # 传入 total_steps 以使注意力模块可以根据训练进度动态调整 α、β、λ1、λ2 等调制因子，实现训练阶段自适应的优化策略
        # 如果不传入则使用硬编码的非线性映射，不依赖于总步数，但同样能提供时间信息（不过早期增长较快，后期趋于饱和）
        total_steps = config['epochs'] * len(train_loader)

        optimizer = MetaAdamW(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay'],
            feature_dim=config['feature_dim'],
            attn_hidden_dim=config['attn_hidden_dim'],
            attn_layers=config['attn_layers'],
            attn_heads=config['attn_heads'],
            meta_update_freq=config['meta_update_freq'],
            meta_lr=config['meta_lr'],
            total_steps=total_steps,
            model=model,                                    # 传入模型实例
            group_strategy=config['group_strategy'],
            feature_version = config['feature_version'],
            include_time_step=config['include_time_step'],
            group_embed_dim =config['group_embed_dim'],
            use_v_norms=config['use_v_norms'],
            use_feature_gating=config['use_feature_gating'],
            feature_gating_sparsity=config['feature_gating_sparsity'],
            meta_objective=config['meta_objective'],
            val_loader=val_loader,                          # 传入验证集加载器（需要提前定义）
            aux_loss_fixed_weights=config['aux_loss_fixed_weights'],
            use_huw=config['use_huw'],
            task_types=config['task_types'],
            huw_priorities=config['huw_priorities'],
        )
        hyperadamw = True

    # 日志
    log_dir = './logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'{optimizer_type}_cifar10_seed{config["seed"]}.csv').replace('\\', '/')
    with open(log_file, 'w') as f:
        f.write('epoch,train_loss,train_acc,val_loss,val_acc,aux_loss,time\n')

    # 早停
    early_stopping = EarlyStopping(config['patience'], config['min_delta'], config['restore_best_weights'])
    best_val_acc = 0.0

    # 训练
    for epoch in range(1, config['epochs'] + 1):
        start_time = time.time()

        train_loss, train_acc, train_aux_loss = train_epoch(
            model, device, train_loader, optimizer, epoch,
            hyperadamw=hyperadamw,
            meta_update_freq=config['meta_update_freq'] if hyperadamw else 10
        )
        val_loss, val_acc = validate(model, device, val_loader)

        # 控制台日志
        epoch_time = time.time() - start_time
        logger.info(f'Epoch {epoch:2d} | Time: {epoch_time:.1f}s | '
                    f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | '
                    f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | '
                    f'Aux Loss: {train_aux_loss:.4f}')

        # 保存最佳模型（可选）
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'./logs/best_model_{optimizer_type}_cifar10.pt')

        # 写入日志
        with open(log_file, 'a') as f:
            f.write(f'{epoch},{train_loss:.4f},{train_acc:.4f},{val_loss:.4f},{val_acc:.4f},'
                    f'{train_aux_loss:.4f},{epoch_time:.2f}\n')

        # 早停检查
        if early_stopping(val_loss, model):
            logger.warning(f"Early stopping triggered at epoch {epoch}")
            break

    logger.success(f'Success, log file: {log_file}')


# ==================== 实验日志 ====================

# AdamW 基线 (patience=2, epochs=50)
# -------------------------------------------------------------------------------------------------------------------------
# Epoch  1 | Time: 2437.7s | Train Loss: 1.5547 | Train Acc: 0.4366 | Val Loss: 1.2832 | Val Acc: 0.5350 | Aux Loss: 0.0000
# Epoch  2 | Time: 2412.6s | Train Loss: 1.1473 | Train Acc: 0.5821 | Val Loss: 0.9065 | Val Acc: 0.6936 | Aux Loss: 0.0000
# Epoch  3 | Time: 2391.4s | Train Loss: 0.9540 | Train Acc: 0.6517 | Val Loss: 0.9082 | Val Acc: 0.7005 | Aux Loss: 0.0000
# Epoch  4 | Time: 2390.4s | Train Loss: 0.8515 | Train Acc: 0.6866 | Val Loss: 0.7059 | Val Acc: 0.7656 | Aux Loss: 0.0000
# Epoch  5 | Time: 2397.2s | Train Loss: 0.7695 | Train Acc: 0.7119 | Val Loss: 0.6436 | Val Acc: 0.7862 | Aux Loss: 0.0000
# Epoch  6 | Time: 2398.2s | Train Loss: 0.7114 | Train Acc: 0.7325 | Val Loss: 0.5577 | Val Acc: 0.8177 | Aux Loss: 0.0000
# Epoch  7 | Time: 2397.8s | Train Loss: 0.6625 | Train Acc: 0.7466 | Val Loss: 0.6195 | Val Acc: 0.8064 | Aux Loss: 0.0000
# Epoch  8 | Time: 2392.1s | Train Loss: 0.6269 | Train Acc: 0.7561 | Val Loss: 0.5059 | Val Acc: 0.8360 | Aux Loss: 0.0000
# Epoch  9 | Time: 2398.7s | Train Loss: 0.5934 | Train Acc: 0.7673 | Val Loss: 0.4884 | Val Acc: 0.8438 | Aux Loss: 0.0000
# Epoch 10 | Time: 2422.5s | Train Loss: 0.5722 | Train Acc: 0.7755 | Val Loss: 0.4058 | Val Acc: 0.8643 | Aux Loss: 0.0000
# Epoch 11 | Time: 2445.9s | Train Loss: 0.5453 | Train Acc: 0.7847 | Val Loss: 0.4411 | Val Acc: 0.8564 | Aux Loss: 0.0000
# Epoch 12 | Time: 2439.8s | Train Loss: 0.5209 | Train Acc: 0.7909 | Val Loss: 0.4084 | Val Acc: 0.8702 | Aux Loss: 0.0000
# Early stopping triggered at epoch 12

# MetaAdamW 最优 (patience=2, epochs=15)
# ------------------------------------------------------------------------------------------------------------------------
# Epoch  1 | Time: 2590.6s | Train Loss: 1.5442 | Train Acc: 0.4422 | Val Loss: 1.5485 | Val Acc: 0.4996 | Aux Loss: 0.3905
# Epoch  2 | Time: 2582.9s | Train Loss: 1.1257 | Train Acc: 0.5913 | Val Loss: 0.9133 | Val Acc: 0.6911 | Aux Loss: 0.1321
# Epoch  3 | Time: 2562.1s | Train Loss: 0.9453 | Train Acc: 0.6540 | Val Loss: 0.8964 | Val Acc: 0.7007 | Aux Loss: 0.0622
# Epoch  4 | Time: 2629.3s | Train Loss: 0.8377 | Train Acc: 0.6915 | Val Loss: 0.8499 | Val Acc: 0.7335 | Aux Loss: 0.3460
# Epoch  5 | Time: 2650.6s | Train Loss: 0.7576 | Train Acc: 0.7182 | Val Loss: 0.6293 | Val Acc: 0.7827 | Aux Loss: 0.2052
# Epoch  6 | Time: 2647.3s | Train Loss: 0.7013 | Train Acc: 0.7355 | Val Loss: 0.5989 | Val Acc: 0.8043 | Aux Loss: 0.2519
# Epoch  7 | Time: 2645.7s | Train Loss: 0.6577 | Train Acc: 0.7475 | Val Loss: 0.5943 | Val Acc: 0.8062 | Aux Loss: 0.1056
# Epoch  8 | Time: 2647.4s | Train Loss: 0.6140 | Train Acc: 0.7630 | Val Loss: 0.5219 | Val Acc: 0.8363 | Aux Loss: 0.1076
# Epoch  9 | Time: 2648.4s | Train Loss: 0.5917 | Train Acc: 0.7681 | Val Loss: 0.4743 | Val Acc: 0.8481 | Aux Loss: 0.1072
# Epoch 10 | Time: 2648.6s | Train Loss: 0.5545 | Train Acc: 0.7804 | Val Loss: 0.4317 | Val Acc: 0.8612 | Aux Loss: 0.1448
# Epoch 11 | Time: 2657.7s | Train Loss: 0.5408 | Train Acc: 0.7850 | Val Loss: 0.4622 | Val Acc: 0.8450 | Aux Loss: 0.1329
# Epoch 12 | Time: 2682.8s | Train Loss: 0.5095 | Train Acc: 0.7940 | Val Loss: 0.3757 | Val Acc: 0.8798 | Aux Loss: 0.0959
# Epoch 13 | Time: 2662.3s | Train Loss: 0.4933 | Train Acc: 0.7984 | Val Loss: 0.4535 | Val Acc: 0.8591 | Aux Loss: 0.1187
# Epoch 14 | Time: 2646.4s | Train Loss: 0.4786 | Train Acc: 0.8040 | Val Loss: 0.3939 | Val Acc: 0.8805 | Aux Loss: 0.2249
# Early stopping triggered at epoch 14

# MetaAdamW 实验终点                                                                                               meta_update_freq    attn_layers     attn_hidden_dim     total_steps     λ1,λ2  warmup_epoch    attn_heads  group_strategy  feature_version feature_dim meta_objective   encoder_dropout combined_weights    HUW inc_time_step use_v_norms feature_gating  epochs    huw_priorities
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Epoch 10 | Time: 2617.1s | Train Loss: 0.5961 | Train Acc: 0.7690 | Val Loss: 0.4400 | Val Acc: 0.8579 | Aux Loss: -0.0859  391              8                  16             yes      auto             1             6    fine_grained            basic           6       combined               0.0             None    Yes          True        True          False      12   [1.0, 1.0, 1.0]
# Epoch  7 | Time: 2652.2s | Train Loss: 0.6922 | Train Acc: 0.7377 | Val Loss: 0.5049 | Val Acc: 0.8297 | Aux Loss: 0.1651   190              8                  16             yes      auto             1             6    fine_grained            basic           6       combined               0.0             None    Yes          True        True           True      12   [1.0, 1.0, 1.0]
# Epoch  8 | Time: 2594.7s | Train Loss: 0.6497 | Train Acc: 0.7514 | Val Loss: 0.5181 | Val Acc: 0.8304 | Aux Loss: 0.2740   190              8                  16             yes      auto             1             6    fine_grained            basic           6       gradient               0.0             None     No          True        True          False      12   [1.0, 1.0, 1.0]
# Epoch 12 | Time: 2872.7s | Train Loss: 0.5327 | Train Acc: 0.7880 | Val Loss: 0.4387 | Val Acc: 0.8602 | Aux Loss: 2.2407   190             32                  64             yes      auto             1             6    fine_grained            basic           6       combined               0.0             None    Yes          True        True           True      12   [1.0, 1.0, 1.0]
# Epoch 10 | Time: 2559.7s | Train Loss: 0.5967 | Train Acc: 0.7690 | Val Loss: 0.4429 | Val Acc: 0.8544 | Aux Loss: 0.0053   391              8                  16             yes      auto             1             6    fine_grained            basic           6       combined               0.0             None    Yes          True        True          False      12   [1.0, 2.0, 3.0]
# Epoch 12 | Time: 2710.6s | Train Loss: 0.5402 | Train Acc: 0.7856 | Val Loss: 0.4361 | Val Acc: 0.8574 | Aux Loss: 0.0612   391              8                  16             yes      auto             1             6    fine_grained            basic           6       combined               0.0             None    Yes          True        True          False      12   [3.0, 2.0, 1.0]
# Epoch  7 | Time: 2720.7s | Train Loss: 0.6551 | Train Acc: 0.7500 | Val Loss: 0.4636 | Val Acc: 0.8448 | Aux Loss: 0.8337   190             32                  64             yes      auto             1             6    fine_grained            basic           6       combined               0.0             None    Yes          True        True          False      12   [3.0, 2.0, 1.0]
# Epoch 14 | Time: 2646.4s | Train Loss: 0.4786 | Train Acc: 0.8040 | Val Loss: 0.3939 | Val Acc: 0.8805 | Aux Loss: 0.2249   190             64                  64             yes      auto             1             6    fine_grained            basic           6       combined               0.0             None    Yes          True        True          False      15   [5.0, 2.0, 1.0]
# Epoch 11 | Time: 2651.0s | Train Loss: 0.5409 | Train Acc: 0.7841 | Val Loss: 0.3658 | Val Acc: 0.8776 | Aux Loss: 0.2780   391             64                  64             yes      auto             1             6    fine_grained            basic           6       combined               0.0             None    Yes          True        True          False      15   [5.0, 2.0, 1.0]
# Epoch  8 | Time: 2841.0s | Train Loss: 0.6211 | Train Acc: 0.7594 | Val Loss: 0.4231 | Val Acc: 0.8600 | Aux Loss: 0.2589   391             64                  64             yes      auto             1             6    fine_grained            basic           6       combined               0.0             None    Yes          True        True          False      15   [5.0, 1.0, 2.0]

