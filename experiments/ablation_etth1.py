"""Time Series Forecasting with Transformer 消融实验 (ETT 数据集)"""

import os
import time
import math
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from loguru import logger
import requests

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

    # 时间序列数据特定配置
    'dataset': 'ETTh1',              # 可选: ETTh1, ETTh2, ETTm1, ETTm2
    'data_path': './data/ETT-small/', # 数据集存放路径
    'seq_len': 96,                   # 输入窗口长度 (常用 96, 192, 336, 720)
    'pred_len': 1,                   # 预测步长 (单步预测)
    'batch_size': 64,
    'val_ratio': 0.2,               # 验证集比例

    # Transformer 模型参数
    'd_model': 128,
    'nhead': 4,
    'num_layers': 2,
    'dropout': 0.2,

    # 优化器通用设置
    'lr': 5e-4,
    'weight_decay': 0.01,

    # MetaAdamW 专属设置
    'meta_update_freq': 217,            # 元更新频率（步数），217 steps/epoch
    'feature_dim': 6,                   # 每个参数组的特征维度（在 'enhanced' 时需设置 16 以确保足够容纳）；若为 None 则自动计算有效特征维度
    'attn_hidden_dim': 16,              # 注意力MLP隐藏维度 16
    'attn_layers': 8,                   # 注意力层数 2
    'attn_heads': 6,                    # 注意力头数
    'meta_lr': 1e-4,                    # 元学习的学习率
    'group_strategy': 'fine_grained',   # 是否启用精细分组：'original' 或 'fine_grained'
    # basic 核心：梯度范数、动量范数、参数范数、梯度与动量的余弦相似度
    # 特征表达： 'basic', 'norm_basic', 'basic_plus', 'norm_basic_plus', 'enhanced'
    'feature_version': 'basic',         # basic: 4 or+1 or+1, basic_plus: 4+4 or+1 or+2, enhanced: 9 or+1 + group_embed_dim
    'use_v_norms': True,                # basic 相关特征中是否启用 二阶动量范数 特征
    'use_feature_gating': True,        # 是否启用 特征门控 机制
    'feature_gating_sparsity': 1e-4,    # 启用特征门控时的正则化强度
    'include_time_step': True,          # 特征中是否包含时间步
    'group_embed_dim': 4,               # 组嵌入向量的维度（仅在 'enhanced' 时使用）
    # 分别对应：梯度方向（使元学习更新的方向更优），损失下降（让模型泛化更好），泛化差距（减小过拟合），三者混合
    'meta_objective': 'combined',       # 元学习目标：'gradient', 'loss_decrease', 'gen_gap', 'combined'

    # 元学习目标为 'combined' 时：辅助损失组合方案
    'aux_loss_fixed_weights': [1.0, 0.3, 0.3],   # 固定权重，若为 None 则使用简单相加
    'use_huw': True,                             # 是否启用同调不确定性加权（优先级高于固定权重）
    'task_types': ['regression', 'regression', 'regression'],
    'huw_priorities': [1.0, 1.0, 1.0],           # HUW 业务优先级，默认等权重

    # 训练设置
    'epochs': 8,
    'log_interval': 10,
    # 一般3轮时主模型大致可以收敛到较平坦区域，然后再启用注意力模块，这样可以避免早期的不稳定影响后续元学习
    'warmup_epoch': 1,                           # 在主模型训练到 第几轮 时，优化器才启动元学习

    # 早停设置
    'patience': 2,
    'min_delta': 1e-4,
    'restore_best_weights': True,
}


# 定义损失函数闭包（用于元学习）
def ts_loss_fn(model, batch):
    """统一损失函数接口，用于元学习"""
    x, y = batch
    pred = model(x)
    return F.mse_loss(pred, y)


# ==================== 数据集准备 (ETT 自动下载) ====================
def download_ett(data_path, dataset_name):
    """从官方仓库下载 ETT 数据集（如果本地不存在）"""
    os.makedirs(data_path, exist_ok=True)
    file_path = os.path.join(data_path, f"{dataset_name}.csv")
    if os.path.exists(file_path):
        logger.info(f"数据集已存在: {file_path}")
        return file_path

    # 官方数据源 URL
    url = f"https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/{dataset_name}.csv"
    logger.info(f"正在从 {url} 下载 {dataset_name}.csv ...")
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        with open(file_path, 'wb') as f:
            f.write(r.content)
        logger.success(f"下载完成: {file_path}")
        return file_path
    except Exception as e:
        raise RuntimeError(f"自动下载失败，请手动下载 {dataset_name}.csv 并放置于 {data_path} 目录下。错误: {e}")


def get_dataloaders(data_path, dataset_name, seq_len, pred_len, val_ratio):
    """
    加载 ETT 数据集，并划分训练/验证。
    返回 (train_loader, val_loader, input_dim, output_dim)
    """
    file_path = download_ett(data_path, dataset_name)
    df = pd.read_csv(file_path)

    # 数据预处理：取 'OT' 列（油温）作为目标变量，也可以使用所有特征，但为简化，这里只使用 OT
    # 若需使用全部特征，可将特征列设为所有数值列（除去日期）
    # 这里为轻量级实验，使用单变量 'OT'
    data = df['OT'].values.astype(np.float32)

    # 标准化
    mean = np.mean(data)
    std = np.std(data)
    data = (data - mean) / std

    # 创建样本 (seq_len 输入，pred_len 输出)
    inputs, targets = [], []
    total_len = len(data)
    for i in range(total_len - seq_len - pred_len + 1):
        inputs.append(data[i:i+seq_len])
        targets.append(data[i+seq_len:i+seq_len+pred_len])
    inputs = np.array(inputs)
    targets = np.array(targets)

    # 划分训练/验证 (按时间顺序，前80%训练，后20%验证)
    split_idx = int(len(inputs) * (1 - val_ratio))
    train_inputs, val_inputs = inputs[:split_idx], inputs[split_idx:]
    train_targets, val_targets = targets[:split_idx], targets[split_idx:]

    class ETTDataset(Dataset):
        def __init__(self, x, y):
            self.x = torch.tensor(x, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32)
        def __len__(self):
            return len(self.x)
        def __getitem__(self, idx):
            return self.x[idx], self.y[idx]

    train_dataset = ETTDataset(train_inputs, train_targets)
    val_dataset = ETTDataset(val_inputs, val_targets)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    logger.info(f"训练样本数: {len(train_dataset)}, 验证样本数: {len(val_dataset)}")
    return train_loader, val_loader


# ==================== Transformer 模型 ====================
class PositionalEncoding(nn.Module):
    """位置编码（正弦余弦）"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1)]


class TimeSeriesTransformer(nn.Module):
    """
    基于 Transformer 编码器的时间序列预测模型。
    输入形状: (batch, seq_len)
    输出形状: (batch, pred_len)
    """
    def __init__(self, seq_len, pred_len, d_model=128, nhead=4, num_layers=2, dropout=0.2):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        # 输入投影：将单变量值映射到 d_model 维
        self.input_proj = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # 输出投影：取最后一个时间步的输出，映射到 pred_len
        self.output_proj = nn.Linear(d_model, pred_len)

        # 初始化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x):
        # x: (batch, seq_len)
        batch_size = x.size(0)
        # 增加特征维度 (batch, seq_len, 1)
        x = x.unsqueeze(-1)
        # 投影到 d_model
        x = self.input_proj(x)                     # (batch, seq_len, d_model)
        # 位置编码
        x = self.pos_encoder(x)                    # (batch, seq_len, d_model)
        # Transformer 编码器
        x = self.transformer_encoder(x)            # (batch, seq_len, d_model)
        # 取最后一个时间步的输出
        last = x[:, -1, :]                         # (batch, d_model)
        out = self.output_proj(last)               # (batch, pred_len)
        return out


# ==================== 训练函数 ====================
def train_epoch(model, device, train_loader, optimizer, epoch,
                hyperadamw=False, meta_update_freq=None):
    """
    训练一个 epoch，返回 平均损失, 平均辅助损失 等
    """
    model.train()
    total_loss = 0.0
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
        inputs, targets = inputs.to(device), targets.to(device)  # targets: (batch, pred_len)

        optimizer.zero_grad()
        pred = model(inputs)
        loss = F.mse_loss(pred, targets)
        loss.backward()
        optimizer.step()

        # 统计
        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
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
                    ts_loss_fn
                )
                aux_loss_total += aux_loss
                aux_loss_count += 1

        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'aux_loss': f'{aux_loss:.4f}'
        })

    avg_loss = total_loss / total_samples
    avg_aux_loss = aux_loss_total / aux_loss_count if aux_loss_count > 0 else 0.0
    return avg_loss, avg_aux_loss


def validate(model, device, val_loader):
    """
    验证，返回 (平均损失)
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            pred = model(inputs)
            loss = F.mse_loss(pred, targets, reduction='sum')
            total_loss += loss.item()
            total_samples += inputs.size(0)
    avg_loss = total_loss / total_samples
    return avg_loss


# ==================== 主程序 ====================
def train_etth1():
    # 配置优化器
    optimizer_type = config['optimizer_type']

    # 日志
    device = torch.device(config['device'])
    logger.info(f"Device: {device}")
    logger.info(f"Optimizer: {optimizer_type}")

    # 加载数据
    train_loader, val_loader = get_dataloaders(
        data_path=config['data_path'],
        dataset_name=config['dataset'],
        seq_len=config['seq_len'],
        pred_len=config['pred_len'],
        val_ratio=config['val_ratio']
    )

    # 创建模型
    model = TimeSeriesTransformer(
        seq_len=config['seq_len'],
        pred_len=config['pred_len'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(device)

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
    log_file = os.path.join(log_dir, f'{optimizer_type}_etth1_seed{config["seed"]}.csv').replace('\\', '/')
    with open(log_file, 'w') as f:
        f.write('epoch,train_loss,val_loss,aux_loss,time\n')

    # 早停
    early_stopping = EarlyStopping(config['patience'], config['min_delta'], config['restore_best_weights'])
    best_val_loss = float('inf')

    # 训练
    for epoch in range(1, config['epochs'] + 1):
        start_time = time.time()

        train_loss, train_aux_loss = train_epoch(
            model, device, train_loader, optimizer, epoch,
            hyperadamw=hyperadamw,
            meta_update_freq=config['meta_update_freq'] if hyperadamw else 10
        )
        val_loss = validate(model, device, val_loader)

        # 控制台日志
        epoch_time = time.time() - start_time
        logger.info(f'Epoch {epoch:2d} | Time: {epoch_time:.1f}s | '
                    f'Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | '
                    f'Aux Loss: {train_aux_loss:.4f}')

        # 保存最佳模型（可选）
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'./logs/best_model_{optimizer_type}_etth1.pt')

        # 写入日志
        with open(log_file, 'a') as f:
            f.write(f'{epoch},{train_loss:.6f},{val_loss:.6f},'
                    f'{train_aux_loss:.4f},{epoch_time:.2f}\n')

        # 早停检查
        if early_stopping(val_loss, model):
            logger.warning(f"Early stopping triggered at epoch {epoch}")
            break

    logger.success(f'Success, log file: {log_file}')


# ==================== 实验日志 ====================

# AdamW 基线 (patience=2, epochs=50)
# --------------------------------------------------------------------------------------
# Epoch  1 | Time: 96.2s | Train Loss: 0.735171 | Val Loss: 0.024184 | Aux Loss: 0.0000
# Epoch  2 | Time: 94.4s | Train Loss: 0.092831 | Val Loss: 0.011356 | Aux Loss: 0.0000
# Epoch  3 | Time: 93.5s | Train Loss: 0.051318 | Val Loss: 0.021206 | Aux Loss: 0.0000
# Epoch  4 | Time: 92.8s | Train Loss: 0.037612 | Val Loss: 0.006887 | Aux Loss: 0.0000
# Epoch  5 | Time: 93.5s | Train Loss: 0.028464 | Val Loss: 0.006249 | Aux Loss: 0.0000
# Epoch  6 | Time: 92.9s | Train Loss: 0.025131 | Val Loss: 0.006147 | Aux Loss: 0.0000
# Epoch  7 | Time: 91.6s | Train Loss: 0.022062 | Val Loss: 0.008885 | Aux Loss: 0.0000
# Epoch  8 | Time: 92.1s | Train Loss: 0.020110 | Val Loss: 0.008166 | Aux Loss: 0.0000
# Early stopping triggered at epoch 8

# MetaAdamW 最优 (patience=2, epochs=8)
# --------------------------------------------------------------------------------------
# Epoch  1 | Time: 99.2s | Train Loss: 0.495220 | Val Loss: 0.027751 | Aux Loss: -0.3878
# Epoch  2 | Time: 98.7s | Train Loss: 0.104259 | Val Loss: 0.010729 | Aux Loss: 3.4421
# Epoch  3 | Time: 98.3s | Train Loss: 0.058425 | Val Loss: 0.007670 | Aux Loss: 0.2448
# Epoch  4 | Time: 98.5s | Train Loss: 0.041161 | Val Loss: 0.006034 | Aux Loss: 0.3630
# Epoch  5 | Time: 101.3s | Train Loss: 0.031901 | Val Loss: 0.005885 | Aux Loss: -0.2487
# Epoch  6 | Time: 99.0s | Train Loss: 0.027565 | Val Loss: 0.006856 | Aux Loss: -0.6535
# Epoch  7 | Time: 98.4s | Train Loss: 0.024845 | Val Loss: 0.006251 | Aux Loss: 0.5504
# Early stopping triggered at epoch 7

# MetaAdamW 实验终点                                                              meta_update_freq    attn_layers     attn_hidden_dim     total_steps     λ1,λ2  warmup_epoch    attn_heads  group_strategy  feature_version feature_dim meta_objective   encoder_dropout combined_weights    HUW inc_time_step use_v_norms feature_gating  epochs    huw_priorities
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Epoch  5 | Time: 100.1s | Train Loss: 0.031924 | Val Loss: 0.005932 | Aux Loss: -0.2228    217              8                  16             yes      auto             1             6    fine_grained            basic           6       combined               0.0             None    Yes          True        True          False      12    [1.0, 1.0, 1.0]
# Epoch  5 | Time: 100.9s | Train Loss: 0.031850 | Val Loss: 0.005940 | Aux Loss: -0.0253    217              8                  16             yes      auto             1             6    fine_grained            basic           6       combined               0.0             None    Yes          True        True           True       5    [1.0, 1.0, 1.0]
# Epoch  5 | Time: 101.3s | Train Loss: 0.031901 | Val Loss: 0.005885 | Aux Loss: -0.2487    217              8                  16             yes      auto             1             6    fine_grained            basic           6       combined               0.0             None    Yes          True        True           True       8    [1.0, 1.0, 1.0]

