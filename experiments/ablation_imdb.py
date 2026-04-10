"""LSTM on IMDB 情感分类消融实验"""

import os
import time
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
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

    # IMDB 数据设置
    'max_len': 256,         # 序列最大长度
    'batch_size': 64,
    'vocab_size': 25000,    # 词汇表大小
    'min_freq': 5,          # 最小词频

    # LSTM 模型参数
    'embed_dim': 32,
    'hidden_dim': 64,
    'num_layers': 2,
    'dropout': 0.2,

    # 优化器通用设置
    'lr': 5e-4,
    'weight_decay': 0.01,

    # MetaAdamW 专属设置
    'meta_update_freq': 39,            # 元更新频率（步数），391 steps/epoch
    'feature_dim': 11,                   # 每个参数组的特征维度（在 'enhanced' 时需设置 16 以确保足够容纳）；若为 None 则自动计算有效特征维度
    'attn_hidden_dim': 64,              # 注意力MLP隐藏维度 16
    'attn_layers': 64,                   # 注意力层数 8
    'attn_heads': 11,                    # 注意力头数
    'meta_lr': 1e-4,                    # 元学习的学习率
    'group_strategy': 'fine_grained',   # 是否启用精细分组：'original' 或 'fine_grained'
    # basic 核心：梯度范数、动量范数、参数范数、梯度与动量的余弦相似度
    # 特征表达： 'basic', 'norm_basic', 'basic_plus', 'norm_basic_plus', 'enhanced'
    'feature_version': 'basic_plus',         # basic: 4 or+1 or+1, basic_plus: 4+4 or+1 or+2, enhanced: 9 or+1 + group_embed_dim
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
    'huw_priorities': [1.0, 3.0, 5.0],           # HUW 业务优先级，默认等权重

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


# 损失函数闭包
def clf_loss_fn(model, batch):
    """统一损失函数接口，用于元学习"""
    x, y = batch
    logits = model(x)
    return F.cross_entropy(logits, y)


# ==================== LSTM 情感分类模型 ====================
class LSTMSentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=2, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, 2)   # 输出2个logits
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        emb = self.embedding(x)                 # (batch, seq_len, embed_dim)
        lstm_out, (hidden, cell) = self.lstm(emb)
        # 取最后一个时间步的隐藏状态（双向拼接）
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)  # (batch, hidden_dim*2)
        hidden = self.dropout(hidden)
        logits = self.fc(hidden)                # (batch, 2)
        return logits


# ==================== 数据集准备 ====================
def load_imdb():
    """加载 IMDB 数据集（使用 torchtext）"""
    try:
        from torchtext.datasets import IMDB
        from torchtext.data.utils import get_tokenizer
        from torchtext.vocab import build_vocab_from_iterator
    except ImportError:
        raise ImportError("需要安装 torchtext: pip install torchtext")

    # 加载数据集（返回迭代器）
    train_iter, test_iter = IMDB(root='./data', split=('train', 'test'))

    # 转换为列表
    train_list = list(train_iter)
    test_list = list(test_iter)

    # 分词器（使用基础空格分词）
    tokenizer = get_tokenizer('basic_english')

    # 构建词汇表
    def yield_tokens(data_iter):
        for label, text in data_iter:
            yield tokenizer(text)

    vocab = build_vocab_from_iterator(
        yield_tokens(train_list),
        specials=['<pad>', '<unk>'],
        max_tokens=config['vocab_size'],
        min_freq=config['min_freq']
    )
    vocab.set_default_index(vocab['<unk>'])
    config['vocab_size'] = len(vocab)  # 更新实际词汇表大小
    logger.info(f"词汇表大小: {len(vocab)}")

    return train_list, test_list, vocab, tokenizer


def encode_text(text, vocab, tokenizer, max_len):
    """将文本编码为 token ids，并填充/截断到 max_len"""
    tokens = tokenizer(text)
    ids = [vocab[token] for token in tokens]
    if len(ids) > max_len:
        ids = ids[:max_len]
    else:
        ids = ids + [vocab['<pad>']] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


class IMDBDataset(Dataset):
    def __init__(self, data_list, vocab, tokenizer, max_len):
        self.data = data_list
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label, text = self.data[idx]
        # 兼容处理标签：torchtext 新版本返回 int (1: neg, 2: pos)，旧版本返回 str
        if isinstance(label, str):
            label = 1 if label == 'pos' else 0
        elif isinstance(label, int):
            # 假设 torchtext 0.12+ 返回 1 和 2，需要映射到 0 和 1
            if label == 1:
                label = 0  # neg
            elif label == 2:
                label = 1  # pos
            else:
                label = label  # 已经是 0/1
        else:
            label = int(label)
        input_ids = encode_text(text, self.vocab, self.tokenizer, self.max_len)
        return input_ids, torch.tensor(label, dtype=torch.long)


def get_dataloaders(batch_size, max_len):
    train_list, test_list, vocab, tokenizer = load_imdb()
    train_dataset = IMDBDataset(train_list, vocab, tokenizer, max_len)
    test_dataset = IMDBDataset(test_list, vocab, tokenizer, max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    logger.info(f"训练样本数: {len(train_dataset)}, 测试样本数: {len(test_dataset)}")
    return train_loader, test_loader


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
        logits = model(inputs)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()

        # 统计
        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        _, pred = logits.max(1)
        total_correct += pred.eq(targets).sum().item()
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
                    clf_loss_fn
                )
                aux_loss_total += aux_loss
                aux_loss_count += 1

        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{total_correct / total_samples:.4f}',
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
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss = F.cross_entropy(logits, targets, reduction='sum')
            total_loss += loss.item()
            _, pred = logits.max(1)
            total_correct += pred.eq(targets).sum().item()
            total_samples += inputs.size(0)
    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


# ==================== 主程序 ====================
def train_imdb():
    # 配置优化器
    optimizer_type = config['optimizer_type']

    # 日志
    device = torch.device(config['device'])
    logger.info(f"Device: {device}")
    logger.info(f"Optimizer: {optimizer_type}")

    # 加载数据
    train_loader, val_loader = get_dataloaders(
        batch_size=config['batch_size'],
        max_len=config['max_len']
    )

    # 创建模型
    model = LSTMSentimentClassifier(
        vocab_size=config['vocab_size'],
        embed_dim=config['embed_dim'],
        hidden_dim=config['hidden_dim'],
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
    log_file = os.path.join(log_dir, f'{optimizer_type}_imdb_seed{config["seed"]}.csv').replace('\\', '/')
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
            torch.save(model.state_dict(), f'./logs/best_model_{optimizer_type}_imdb.pt')

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
# Epoch  1 | Time: 706.5s | Train Loss: 0.6777 | Train Acc: 0.5610 | Val Loss: 0.6606 | Val Acc: 0.5969 | Aux Loss: 0.0000
# Epoch  2 | Time: 343.9s | Train Loss: 0.6173 | Train Acc: 0.6605 | Val Loss: 0.6067 | Val Acc: 0.6618 | Aux Loss: 0.0000
# Epoch  3 | Time: 164.8s | Train Loss: 0.5995 | Train Acc: 0.6778 | Val Loss: 0.6001 | Val Acc: 0.6762 | Aux Loss: 0.0000
# Epoch  4 | Time: 161.2s | Train Loss: 0.5208 | Train Acc: 0.7475 | Val Loss: 0.5306 | Val Acc: 0.7421 | Aux Loss: 0.0000
# Epoch  5 | Time: 161.7s | Train Loss: 0.4799 | Train Acc: 0.7792 | Val Loss: 0.5439 | Val Acc: 0.7453 | Aux Loss: 0.0000
# Epoch  6 | Time: 161.9s | Train Loss: 0.5066 | Train Acc: 0.7549 | Val Loss: 0.5319 | Val Acc: 0.7443 | Aux Loss: 0.0000
# Early stopping triggered at epoch 6

# MetaAdamW 最优 (patience=2, epochs=15)
# ------------------------------------------------------------------------------------------------------------------------
# Epoch  1 | Time: 1002.2s | Train Loss: 0.6859 | Train Acc: 0.5422 | Val Loss: 0.6765 | Val Acc: 0.5708 | Aux Loss: 0.1591
# Epoch  2 | Time: 679.1s | Train Loss: 0.6453 | Train Acc: 0.6270 | Val Loss: 0.6437 | Val Acc: 0.6234 | Aux Loss: 0.1635
# Epoch  3 | Time: 322.5s | Train Loss: 0.5912 | Train Acc: 0.6874 | Val Loss: 0.5997 | Val Acc: 0.6912 | Aux Loss: 0.2574
# Epoch  4 | Time: 292.4s | Train Loss: 0.5350 | Train Acc: 0.7381 | Val Loss: 0.5329 | Val Acc: 0.7382 | Aux Loss: 0.2694
# Epoch  5 | Time: 265.3s | Train Loss: 0.4884 | Train Acc: 0.7674 | Val Loss: 0.5203 | Val Acc: 0.7395 | Aux Loss: 0.3183
# Epoch  6 | Time: 264.9s | Train Loss: 0.4404 | Train Acc: 0.8017 | Val Loss: 0.4704 | Val Acc: 0.7751 | Aux Loss: 0.2567
# Epoch  7 | Time: 260.4s | Train Loss: 0.4047 | Train Acc: 0.8226 | Val Loss: 0.4518 | Val Acc: 0.7880 | Aux Loss: 0.1553
# Epoch  8 | Time: 260.8s | Train Loss: 0.3782 | Train Acc: 0.8364 | Val Loss: 0.4299 | Val Acc: 0.8107 | Aux Loss: 0.1139
# Epoch  9 | Time: 260.4s | Train Loss: 0.3550 | Train Acc: 0.8510 | Val Loss: 0.4543 | Val Acc: 0.8052 | Aux Loss: 0.2408
# Epoch 10 | Time: 259.0s | Train Loss: 0.3372 | Train Acc: 0.8591 | Val Loss: 0.4201 | Val Acc: 0.8183 | Aux Loss: 0.1215
# Epoch 11 | Time: 254.3s | Train Loss: 0.3147 | Train Acc: 0.8702 | Val Loss: 0.4009 | Val Acc: 0.8215 | Aux Loss: 0.1794
# Epoch 12 | Time: 254.3s | Train Loss: 0.2956 | Train Acc: 0.8809 | Val Loss: 0.4045 | Val Acc: 0.8279 | Aux Loss: 0.1465
# Epoch 13 | Time: 257.6s | Train Loss: 0.2832 | Train Acc: 0.8873 | Val Loss: 0.4075 | Val Acc: 0.8223 | Aux Loss: 0.1294
# Early stopping triggered at epoch 13

# MetaAdamW 实验终点                                                                                               meta_update_freq    attn_layers     attn_hidden_dim     total_steps     λ1,λ2  warmup_epoch    attn_heads  group_strategy  feature_version feature_dim meta_objective   encoder_dropout combined_weights    HUW inc_time_step use_v_norms feature_gating  epochs    huw_priorities
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Epoch  3 | Time: 290.4s | Train Loss: 0.5897 | Train Acc: 0.6852 | Val Loss: 0.5970 | Val Acc: 0.6767 | Aux Loss: 0.2057    391             64                  64             yes      auto             1             6    fine_grained            basic           6       combined               0.0             None    Yes          True        True          False      15   [5.0, 1.0, 2.0]
# Epoch  8 | Time: 306.1s | Train Loss: 0.3751 | Train Acc: 0.8368 | Val Loss: 0.4665 | Val Acc: 0.7938 | Aux Loss: 0.5762     39             64                  64             yes      auto             1             6    fine_grained            basic           6       combined               0.0             None    Yes          True        True          False      15   [3.0, 5.0, 1.0]
# Epoch  8 | Time: 278.0s | Train Loss: 0.4018 | Train Acc: 0.8246 | Val Loss: 0.4738 | Val Acc: 0.7811 | Aux Loss: 0.5142     39             64                  64             yes      auto             1             6    fine_grained            basic           6       combined               0.0             None    Yes          True        True          False      15   [1.0, 3.0, 5.0]
# Epoch 10 | Time: 284.8s | Train Loss: 0.3372 | Train Acc: 0.8591 | Val Loss: 0.4201 | Val Acc: 0.8183 | Aux Loss: 0.1215     39             64                  64             yes      auto             1            11    fine_grained       basic_plus          11       combined               0.0             None    Yes          True        True           True      10   [1.0, 3.0, 5.0]
# Epoch 12 | Time: 254.3s | Train Loss: 0.2956 | Train Acc: 0.8809 | Val Loss: 0.4045 | Val Acc: 0.8279 | Aux Loss: 0.1465     39             64                  64             yes      auto             1            11    fine_grained       basic_plus          11       combined               0.0             None    Yes          True        True           True      15   [1.0, 3.0, 5.0]

