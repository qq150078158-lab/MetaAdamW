"""MiniGPT on WikiText-2 消融实验"""

import os
import time
import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from loguru import logger
from collections import Counter

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

    # MiniGPT 数据设置
    'dataset': 'wikitext-2',    # 使用 WikiText-2
    'block_size': 128,          # 每个训练样本的上下文长度
    'batch_size': 64,
    'vocab_size': 10000,        # 词汇表大小，限制词汇量
    'min_freq': 2,              # 词汇表最小词频

    # MiniGPT 特定配置
    'd_model': 128,
    'nhead': 4,
    'num_layers': 2,
    'dropout': 0.2,

    # 优化器通用设置
    'lr': 5e-4,
    'weight_decay': 0.01,

    # MetaAdamW 专属设置
    'meta_update_freq': 123,            # 元更新频率（步数），123 steps/epoch
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
    'huw_priorities': [3.0, 5.0, 1.0],           # HUW 业务优先级，默认等权重

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


# 统一接收 (model, batch)，在内部解包
def nlp_loss_fn(eval_model, batch):
    x, y = batch
    _, loss = eval_model(x, y)
    return loss


# ==================== 轻量级 GPT 语言模型 ====================
class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2,
                 block_size=128, dropout=0.2):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(block_size, d_model)
        # 使用 TransformerDecoderLayer 需要提供掩码，这里直接用 TransformerEncoderLayer + 因果掩码
        # 但 torch.nn.TransformerEncoderLayer 支持 src_mask，我们手动构造上三角掩码
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=4*d_model, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

        # 权重初始化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        # idx: (batch, seq_len)
        batch_size, seq_len = idx.shape
        assert seq_len <= self.block_size

        # token + position embeddings
        tok_emb = self.token_embedding(idx)                     # (B, T, C)
        pos = torch.arange(0, seq_len, device=idx.device).unsqueeze(0)  # (1, T)
        pos_emb = self.position_embedding(pos)                  # (1, T, C)
        x = tok_emb + pos_emb

        # 创建因果掩码 (上三角)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=idx.device), diagonal=1).bool()
        x = self.transformer(x, mask=mask)                      # (B, T, C)
        x = self.ln_f(x)
        logits = self.lm_head(x)                                # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # 计算交叉熵损失，忽略 pad token (假设 0 为 pad)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0)
        return logits, loss


# ==================== 数据集准备 ====================
def load_wikitext2(raw_data_path=None):
    """
    加载 WikiText-2 数据集，返回训练和验证文本行列表。
    如果本地无文件，将自动下载（需安装 datasets 库）。
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("请安装 datasets 库: pip install datasets")
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
    train_texts = dataset['train']['text']
    val_texts = dataset['validation']['text']
    # 过滤空行
    train_texts = [line for line in train_texts if line.strip()]
    val_texts = [line for line in val_texts if line.strip()]
    return train_texts, val_texts

def build_vocab(texts, vocab_size, min_freq=2):
    """
    从文本构建词汇表，返回 word2idx 和 idx2word。
    使用简单的空格分词，并添加特殊符号: <PAD>=0, <UNK>=1
    """
    counter = Counter()
    for line in texts:
        tokens = line.strip().split()
        counter.update(tokens)
    # 取频率最高的词
    most_common = counter.most_common(vocab_size - 2)  # 留出 <PAD> 和 <UNK>
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, _ in most_common:
        if word not in vocab:
            vocab[word] = len(vocab)
    return vocab

def encode_text(text, vocab, max_len=None):
    tokens = text.strip().split()
    ids = [vocab.get(token, 1) for token in tokens]   # 1 = <UNK>
    if max_len is not None and len(ids) > max_len:
        ids = ids[:max_len]
    return ids

class LanguageModelingDataset(Dataset):
    def __init__(self, texts, vocab, block_size):
        self.data = []
        for text in texts:
            ids = encode_text(text, vocab)
            # 将长文本切分成 block_size 长度的块（步长为 block_size，无重叠）
            for i in range(0, len(ids) - block_size + 1, block_size):
                chunk = ids[i:i+block_size]
                self.data.append(chunk)
        # 如果数据量太大，可采样一部分以加快速度（这里全量使用）
        logger.info(f"构建数据集: {len(self.data)} 个样本")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        chunk = torch.tensor(self.data[idx], dtype=torch.long)
        # 输入和输出: 输入为 chunk[:-1]，输出为 chunk[1:]
        # 但为了简化，我们让模型接受完整 chunk，内部自动处理偏移（需注意 padding）
        # 这里直接返回 (input_ids, target_ids) 且长度相同
        input_ids = chunk[:-1]
        target_ids = chunk[1:]
        return input_ids, target_ids

def collate_fn(batch):
    """
    批处理函数：对每个样本的 input_ids 和 target_ids 分别填充到相同长度。
    注意：由于我们使用了 block_size 切块，所有样本长度应相等（=block_size-1），但安全起见仍然填充。
    """
    inputs, targets = zip(*batch)
    # 填充到 batch 内最大长度
    inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    targets = pad_sequence(targets, batch_first=True, padding_value=0)
    return inputs, targets

def get_dataloaders(batch_size, block_size, vocab_size, min_freq):
    # 加载原始文本
    train_texts, val_texts = load_wikitext2()
    # 构建词汇表（仅从训练集）
    vocab = build_vocab(train_texts, vocab_size, min_freq)
    config['vocab_size'] = len(vocab)   # 实际词汇表大小
    logger.info(f"词汇表大小: {len(vocab)}")
    # 创建数据集
    train_dataset = LanguageModelingDataset(train_texts, vocab, block_size)
    val_dataset = LanguageModelingDataset(val_texts, vocab, block_size)
    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader


# ==================== 训练函数 ====================
def train_epoch(model, device, train_loader, optimizer, epoch,
                hyperadamw=False, meta_update_freq=None):
    model.train()
    total_loss = 0.0
    total_tokens = 0
    aux_loss_total = 0.0    # 累加辅助损失
    aux_loss_count = 0      # 辅助损失更新次数
    aux_loss = 0.0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch}', leave=True, ncols=140)

    if hyperadamw and meta_update_freq is not None and meta_update_freq > 0:
        # 创建两个迭代器用于元学习
        train_iter1 = iter(train_loader)
        train_iter2 = iter(train_loader)

    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs = inputs.to(device)      # (B, seq_len)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits, loss = model(inputs, targets)
        loss.backward()
        optimizer.step()

        batch_tokens = (targets != 0).sum().item()   # 忽略 padding
        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens

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
                batch1 = (data1, target1)
                batch2 = (data2, target2)

                aux_loss = optimizer.update_attention(
                    model,
                    batch1,
                    batch2,
                    nlp_loss_fn
                )
                aux_loss_total += aux_loss
                aux_loss_count += 1

        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'ppl': f'{math.exp(loss.item()):.2f}',
            'aux_loss': f'{aux_loss:.4f}'
        })

    avg_loss = total_loss / total_tokens
    avg_ppl = math.exp(avg_loss)
    avg_aux_loss = aux_loss_total / aux_loss_count if aux_loss_count > 0 else 0.0
    return avg_loss, avg_ppl, avg_aux_loss

def validate(model, device, val_loader):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            _, loss = model(inputs, targets)
            batch_tokens = (targets != 0).sum().item()
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens
    avg_loss = total_loss / total_tokens
    avg_ppl = math.exp(avg_loss)
    return avg_loss, avg_ppl


# ==================== 主程序 ====================
def train_wikitext2():
    # 配置优化器
    optimizer_type = config['optimizer_type']

    # 日志
    device = torch.device(config['device'])
    logger.info(f"Device: {device}")
    logger.info(f"Optimizer: {optimizer_type}")

    # 加载数据
    train_loader, val_loader = get_dataloaders(
        batch_size=config['batch_size'],
        block_size=config['block_size'],
        vocab_size=config['vocab_size'],
        min_freq=config['min_freq']
    )

    # 创建模型
    model = MiniGPT(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        block_size=config['block_size'],
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
    log_file = os.path.join(log_dir, f'{optimizer_type}_wikitext2_seed{config["seed"]}.csv').replace('\\', '/')
    with open(log_file, 'w') as f:
        f.write('epoch,train_loss,train_ppl,val_loss,val_ppl,aux_loss,time\n')

    # 早停
    early_stopping = EarlyStopping(config['patience'], config['min_delta'], config['restore_best_weights'])
    best_val_ppl = float('inf')

    # 训练
    for epoch in range(1, config['epochs'] + 1):
        start_time = time.time()

        train_loss, train_ppl, train_aux_loss = train_epoch(
            model, device, train_loader, optimizer, epoch,
            hyperadamw=hyperadamw,
            meta_update_freq=config['meta_update_freq'] if hyperadamw else 10
        )
        val_loss, val_ppl = validate(model, device, val_loader)

        # 控制台日志
        epoch_time = time.time() - start_time
        logger.info(f'Epoch {epoch:2d} | Time: {epoch_time:.1f}s | '
                    f'Train Loss: {train_loss:.4f} | Train PPL: {train_ppl:.2f} | '
                    f'Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f} | '
                    f'Aux Loss: {train_aux_loss:.4f}')

        # 保存最佳模型（可选）
        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            torch.save(model.state_dict(), f'./logs/best_val_ppl_{optimizer_type}_wikitext2.pt')

        # 写入日志
        with open(log_file, 'a') as f:
            f.write(f'{epoch},{train_loss:.4f},{train_ppl:.2f},{val_loss:.4f},{val_ppl:.2f},'
                    f'{train_aux_loss:.4f},{epoch_time:.2f}\n')

        # 早停检查
        if early_stopping(val_loss, model):
            logger.warning(f"Early stopping triggered at epoch {epoch}")
            torch.save(model.state_dict(), f'./logs/best_val_loss_{optimizer_type}_lm.pt')
            break

    logger.success(f'Success, log file: {log_file}')


# ==================== 实验日志 ====================

# AdamW 基线 (patience=2, epochs=50)
# ------------------------------------------------------------------------------------------------------------------------
# Epoch  1 | Time: 347.1s | Train Loss: 6.8863 | Train PPL: 978.75 | Val Loss: 5.9661 | Val PPL: 389.99 | Aux Loss: 0.0000
# Epoch  2 | Time: 347.0s | Train Loss: 5.9024 | Train PPL: 365.91 | Val Loss: 5.5936 | Val PPL: 268.71 | Aux Loss: 0.0000
# Epoch  3 | Time: 343.3s | Train Loss: 5.5802 | Train PPL: 265.12 | Val Loss: 5.3673 | Val PPL: 214.28 | Aux Loss: 0.0000
# Epoch  4 | Time: 346.3s | Train Loss: 5.3475 | Train PPL: 210.09 | Val Loss: 5.2066 | Val PPL: 182.48 | Aux Loss: 0.0000
# Epoch  5 | Time: 346.8s | Train Loss: 5.1624 | Train PPL: 174.59 | Val Loss: 5.0898 | Val PPL: 162.36 | Aux Loss: 0.0000
# Epoch  6 | Time: 340.0s | Train Loss: 5.0085 | Train PPL: 149.68 | Val Loss: 5.0086 | Val PPL: 149.69 | Aux Loss: 0.0000
# Epoch  7 | Time: 341.8s | Train Loss: 4.8749 | Train PPL: 130.96 | Val Loss: 4.9471 | Val PPL: 140.77 | Aux Loss: 0.0000
# Epoch  8 | Time: 341.1s | Train Loss: 4.7568 | Train PPL: 116.37 | Val Loss: 4.8876 | Val PPL: 132.63 | Aux Loss: 0.0000
# Epoch  9 | Time: 341.2s | Train Loss: 4.6481 | Train PPL: 104.39 | Val Loss: 4.8557 | Val PPL: 128.48 | Aux Loss: 0.0000
# Epoch 10 | Time: 343.3s | Train Loss: 4.5507 | Train PPL: 94.69 | Val Loss: 4.8254 | Val PPL: 124.64 | Aux Loss: 0.0000
# Epoch 11 | Time: 349.2s | Train Loss: 4.4610 | Train PPL: 86.58 | Val Loss: 4.8089 | Val PPL: 122.60 | Aux Loss: 0.0000
# Epoch 12 | Time: 346.7s | Train Loss: 4.3787 | Train PPL: 79.74 | Val Loss: 4.8068 | Val PPL: 122.33 | Aux Loss: 0.0000
# Epoch 13 | Time: 384.7s | Train Loss: 4.3007 | Train PPL: 73.75 | Val Loss: 4.7914 | Val PPL: 120.47 | Aux Loss: 0.0000
# Epoch 14 | Time: 384.2s | Train Loss: 4.2304 | Train PPL: 68.75 | Val Loss: 4.8050 | Val PPL: 122.12 | Aux Loss: 0.0000
# Epoch 15 | Time: 385.8s | Train Loss: 4.1628 | Train PPL: 64.25 | Val Loss: 4.8049 | Val PPL: 122.11 | Aux Loss: 0.0000
# Early stopping triggered at epoch 15

# MetaAdamW 最优 (patience=2, epochs=15)
# ------------------------------------------------------------------------------------------------------------------------
# Epoch  1 | Time: 365.2s | Train Loss: 6.7087 | Train PPL: 819.49 | Val Loss: 5.8483 | Val PPL: 346.65 | Aux Loss: -0.1824
# Epoch  2 | Time: 358.3s | Train Loss: 5.7728 | Train PPL: 321.44 | Val Loss: 5.4567 | Val PPL: 234.32 | Aux Loss: -0.1209
# Epoch  3 | Time: 359.8s | Train Loss: 5.4264 | Train PPL: 227.33 | Val Loss: 5.2223 | Val PPL: 185.35 | Aux Loss: -0.1466
# Epoch  4 | Time: 357.0s | Train Loss: 5.1769 | Train PPL: 177.13 | Val Loss: 5.0701 | Val PPL: 159.20 | Aux Loss: -0.1360
# Epoch  5 | Time: 354.6s | Train Loss: 4.9813 | Train PPL: 145.66 | Val Loss: 4.9588 | Val PPL: 142.42 | Aux Loss: -0.0430
# Epoch  6 | Time: 352.3s | Train Loss: 4.8182 | Train PPL: 123.74 | Val Loss: 4.8712 | Val PPL: 130.48 | Aux Loss: -0.0535
# Epoch  7 | Time: 352.7s | Train Loss: 4.6802 | Train PPL: 107.80 | Val Loss: 4.8300 | Val PPL: 125.21 | Aux Loss: 0.0349
# Epoch  8 | Time: 357.2s | Train Loss: 4.5569 | Train PPL: 95.29 | Val Loss: 4.7882 | Val PPL: 120.08 | Aux Loss: -0.0046
# Epoch  9 | Time: 365.1s | Train Loss: 4.4489 | Train PPL: 85.53 | Val Loss: 4.7687 | Val PPL: 117.77 | Aux Loss: 0.1117
# Epoch 10 | Time: 362.6s | Train Loss: 4.3507 | Train PPL: 77.53 | Val Loss: 4.7510 | Val PPL: 115.70 | Aux Loss: 0.1332
# Epoch 11 | Time: 354.1s | Train Loss: 4.2616 | Train PPL: 70.92 | Val Loss: 4.7494 | Val PPL: 115.51 | Aux Loss: 0.2414
# Epoch 12 | Time: 354.7s | Train Loss: 4.1787 | Train PPL: 65.28 | Val Loss: 4.7508 | Val PPL: 115.68 | Aux Loss: 0.3683
# Epoch 13 | Time: 351.6s | Train Loss: 4.1029 | Train PPL: 60.52 | Val Loss: 4.7535 | Val PPL: 115.99 | Aux Loss: 0.3683
# Early stopping triggered at epoch 13

# MetaAdamW 实验终点                                                                                              meta_update_freq    attn_layers     attn_hidden_dim     total_steps     λ1,λ2  warmup_epoch    attn_heads  group_strategy  feature_version feature_dim meta_objective   encoder_dropout combined_weights    HUW inc_time_step use_v_norms feature_gating  epochs    huw_priorities
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Epoch 10 | Time: 1480.1s | Train Loss: 4.3853 | Train PPL: 80.26 | Val Loss: 4.7665 | Val PPL: 117.50                        1              2                  16            None       0.5             1             4        original            basic           8       gradient               0.0             None   None          True       False          False      10              None
# Epoch 10 | Time: 452.0s | Train Loss: 4.3883 | Train PPL: 80.51 | Val Loss: 4.7751 | Val PPL: 118.52                        10              2                  16            None       0.5             1             4        original            basic           8       gradient               0.0             None   None          True       False          False      10              None
# Epoch 10 | Time: 450.1s | Train Loss: 4.3729 | Train PPL: 79.27 | Val Loss: 4.7564 | Val PPL: 116.33                        10              8                  16            None       0.5             1             4        original            basic           8       gradient               0.0             None   None          True       False          False      10              None
# Epoch 10 | Time: 526.6s | Train Loss: 4.3606 | Train PPL: 78.30 | Val Loss: 4.7620 | Val PPL: 116.98                        10             16                  16            None       0.5             1             4        original            basic           8       gradient               0.0             None   None          True       False          False      10              None
# Epoch 10 | Time: 466.6s | Train Loss: 4.3735 | Train PPL: 79.32 | Val Loss: 4.7683 | Val PPL: 117.72 | Aux Loss: 0.2022     10              8                  32            None       0.5             1             4        original            basic           8       gradient               0.0             None   None          True       False          False      10              None
# Epoch 10 | Time: 455.3s | Train Loss: 4.3631 | Train PPL: 78.50 | Val Loss: 4.7582 | Val PPL: 116.53 | Aux Loss: 0.4443     10              8                  16            None      auto             1             4        original            basic           8       gradient               0.0             None   None          True       False          False      10              None
# Epoch 10 | Time: 440.8s | Train Loss: 4.3599 | Train PPL: 78.25 | Val Loss: 4.7580 | Val PPL: 116.52 | Aux Loss: 0.4569     10              8                  16             yes      auto             3             4        original            basic           8       gradient               0.0             None   None          True       False          False      10              None
# Epoch 10 | Time: 448.7s | Train Loss: 4.3599 | Train PPL: 78.25 | Val Loss: 4.7580 | Val PPL: 116.52 | Aux Loss: 0.4569     10              8                  16             yes      auto             3             8        original            basic           8       gradient               0.0             None   None          True       False          False      10              None
# Epoch 10 | Time: 401.1s | Train Loss: 4.3599 | Train PPL: 78.25 | Val Loss: 4.7547 | Val PPL: 116.13 | Aux Loss: 0.5025     20              8                  16             yes      auto             3             4        original            basic           8       gradient               0.0             None   None          True       False          False      10              None
# Epoch 10 | Time: 360.9s | Train Loss: 4.3598 | Train PPL: 78.24 | Val Loss: 4.7619 | Val PPL: 116.97 | Aux Loss: 0.4837     50              8                  16             yes      auto             3             4        original            basic           8       gradient               0.0             None   None          True       False          False      10              None
# Epoch 10 | Time: 363.7s | Train Loss: 4.3586 | Train PPL: 78.15 | Val Loss: 4.7553 | Val PPL: 116.19 | Aux Loss: 0.6693    123              8                  16             yes      auto             3             4        original            basic           8       gradient               0.0             None   None          True       False          False      10              None
# Epoch 10 | Time: 401.0s | Train Loss: 4.3666 | Train PPL: 78.78 | Val Loss: 4.7582 | Val PPL: 116.53 | Aux Loss: 0.6083    123              8                  16             yes      auto             1             4        original            basic           8       gradient               0.0             None   None          True       False          False      10              None
# Epoch 10 | Time: 347.4s | Train Loss: 4.3585 | Train PPL: 78.14 | Val Loss: 4.7551 | Val PPL: 116.17 | Aux Loss: 0.0139    123              8                  16             yes      auto             3             4    fine_grained            basic           8       gradient               0.0             None   None          True       False          False      10              None
# Epoch 10 | Time: 349.0s | Train Loss: 4.3663 | Train PPL: 78.75 | Val Loss: 4.7560 | Val PPL: 116.28 | Aux Loss: -0.0360   123              8                  16             yes      auto             1             4    fine_grained            basic           8       gradient               0.0             None   None          True       False          False      10              None
# Epoch 10 | Time: 347.5s | Train Loss: 4.4824 | Train PPL: 88.45 | Val Loss: 4.8003 | Val PPL: 121.55 | Aux Loss: -0.1844   123              8                  16             yes      auto             1             4    fine_grained         enhanced          16       gradient               0.0             None   None          True       False          False      10              None
# Epoch 10 | Time: 346.4s | Train Loss: 4.3638 | Train PPL: 78.56 | Val Loss: 4.7596 | Val PPL: 116.70 | Aux Loss: 0.0245    123              8                  16             yes      auto             1             4    fine_grained            basic           8  loss_decrease               0.0             None   None          True       False          False      10              None
# Epoch 10 | Time: 350.8s | Train Loss: 4.3634 | Train PPL: 78.53 | Val Loss: 4.7576 | Val PPL: 116.47 | Aux Loss: 0.6927    123              8                  16             yes      auto             1             4    fine_grained            basic           8        gen_gap               0.0             None   None          True       False          False      10              None
# Epoch 10 | Time: 361.7s | Train Loss: 4.3612 | Train PPL: 78.35 | Val Loss: 4.7585 | Val PPL: 116.57 | Aux Loss: 0.5920    123              8                  16             yes      auto             1             4    fine_grained            basic           8       combined               0.0  [1.0, 1.0, 1.0]     No          True       False          False      10              None
# Epoch 10 | Time: 436.7s | Train Loss: 4.3722 | Train PPL: 79.22 | Val Loss: 4.7599 | Val PPL: 116.73 | Aux Loss: 0.5342     20              8                  16             yes      auto             1             4    fine_grained            basic           8       combined               0.0  [1.0, 1.0, 1.0]     No          True       False          False      10              None
# Epoch 10 | Time: 361.5s | Train Loss: 4.4745 | Train PPL: 87.75 | Val Loss: 4.7981 | Val PPL: 121.28 | Aux Loss: 0.2835    123              8                  16             yes      auto             1             4    fine_grained         enhanced          16       combined               0.0  [1.0, 1.0, 1.0]     No          True       False          False      10              None
# Epoch 10 | Time: 353.7s | Train Loss: 4.3653 | Train PPL: 78.68 | Val Loss: 4.7613 | Val PPL: 116.89 | Aux Loss: 0.6375    123              8                  16             yes      auto             1             4    fine_grained            basic           8       combined               0.1  [1.0, 1.0, 1.0]     No          True       False          False      10              None
# Epoch 10 | Time: 367.6s | Train Loss: 4.3583 | Train PPL: 78.13 | Val Loss: 4.7589 | Val PPL: 116.62 | Aux Loss: 0.1599    123              8                  16             yes      auto             3             4    fine_grained            basic           8       combined               0.0  [1.0, 0.3, 0.3]     No          True       False          False      10              None
# Epoch 10 | Time: 360.4s | Train Loss: 4.3587 | Train PPL: 78.16 | Val Loss: 4.7599 | Val PPL: 116.73 | Aux Loss: 0.2592    123              8                  16             yes      auto             3             4    fine_grained            basic           8       combined               0.0             None    Yes          True       False          False      10   [1.0, 1.0, 1.0]
# Epoch 10 | Time: 364.6s | Train Loss: 4.3614 | Train PPL: 78.37 | Val Loss: 4.7603 | Val PPL: 116.78 | Aux Loss: 0.3106    123              8                  16             yes      auto             1             4    fine_grained            basic           8       combined               0.0             None    Yes          True       False          False      10   [1.0, 1.0, 1.0]
# Epoch 10 | Time: 377.4s | Train Loss: 4.4740 | Train PPL: 87.70 | Val Loss: 4.7959 | Val PPL: 121.02 | Aux Loss: 0.0234    123              8                  16             yes      auto             1             4    fine_grained         enhanced          16       combined               0.0             None    Yes          True       False          False      10   [1.0, 1.0, 1.0]
# Epoch 10 | Time: 352.6s | Train Loss: 4.3633 | Train PPL: 78.51 | Val Loss: 4.7526 | Val PPL: 115.89 | Aux Loss: 0.1411    123              8                  16             yes      auto             1             2    fine_grained            basic           6       combined               0.0             None    Yes          True       False          False      10   [1.0, 1.0, 1.0]
# Epoch 10 | Time: 359.3s | Train Loss: 4.5236 | Train PPL: 92.17 | Val Loss: 4.8138 | Val PPL: 123.20 | Aux Loss: -0.1636   123              8                  16             yes      auto             1             2    fine_grained         enhanced          14       combined               0.0             None    Yes          True       False          False      10   [1.0, 1.0, 1.0]
# Epoch 10 | Time: 360.4s | Train Loss: 4.3725 | Train PPL: 79.24 | Val Loss: 4.7624 | Val PPL: 117.03 | Aux Loss: 0.0809    123              8                  16             yes      auto             1             1    fine_grained            basic           5       combined               0.0             None    Yes          True       False          False      10   [1.0, 1.0, 1.0]
# Epoch 10 | Time: 355.2s | Train Loss: 4.3634 | Train PPL: 78.53 | Val Loss: 4.7532 | Val PPL: 115.95 | Aux Loss: 0.1345    123              8                  16             yes      auto             1             3    fine_grained            basic           6       combined               0.0             None    Yes          True       False          False      10   [1.0, 1.0, 1.0]
# Epoch 10 | Time: 361.7s | Train Loss: 4.3552 | Train PPL: 77.89 | Val Loss: 4.7517 | Val PPL: 115.78 | Aux Loss: 0.1742    123              8                  16             yes      auto             1             6    fine_grained            basic           6       combined               0.0             None    Yes          True       False          False      10   [1.0, 1.0, 1.0]
# Epoch 10 | Time: 375.3s | Train Loss: 4.3946 | Train PPL: 81.01 | Val Loss: 4.7616 | Val PPL: 116.94 | Aux Loss: 0.2424    123              8                  16             yes      auto             1             6        original            basic           6       combined               0.0             None    Yes          True       False          False      10   [1.0, 1.0, 1.0]
# Epoch 10 | Time: 362.7s | Train Loss: 4.3631 | Train PPL: 78.50 | Val Loss: 4.7564 | Val PPL: 116.32 | Aux Loss: -0.0030   123             16                  64             yes      auto             1             6    fine_grained            basic           6       combined               0.0             None    Yes          True       False          False      10   [1.0, 1.0, 1.0]
# Epoch 10 | Time: 429.0s | Train Loss: 4.3809 | Train PPL: 79.91 | Val Loss: 4.7612 | Val PPL: 116.89 | Aux Loss: 0.0260    123              8                  16             yes      auto             1             4    fine_grained            basic           4       combined               0.0             None    Yes         False       False          False      10   [1.0, 1.0, 1.0]
# Epoch 10 | Time: 363.2s | Train Loss: 4.3716 | Train PPL: 79.17 | Val Loss: 4.7639 | Val PPL: 117.21 | Aux Loss: 0.0773    123              8                  16             yes      auto             1             5    fine_grained            basic           5       combined               0.0             None    Yes          True       False          False      10   [1.0, 1.0, 1.0]
# Epoch 10 | Time: 360.4s | Train Loss: 4.7198 | Train PPL: 112.15 | Val Loss: 4.9280 | Val PPL: 138.11 | Aux Loss: -0.0716  123              8                  16             yes      auto             1             6    fine_grained       norm_basic           6       combined               0.0             None    Yes          True       False          False      10   [1.0, 1.0, 1.0]
# Epoch 10 | Time: 364.4s | Train Loss: 4.9781 | Train PPL: 145.20 | Val Loss: 5.0436 | Val PPL: 155.03 | Aux Loss: -0.2517  123              8                  16             yes      auto             1             9    fine_grained       basic_plus           9       combined               0.0             None    Yes          True       False          False      10   [1.0, 1.0, 1.0]
# Epoch 10 | Time: 364.2s | Train Loss: 4.3552 | Train PPL: 77.89 | Val Loss: 4.7517 | Val PPL: 115.78 | Aux Loss: 0.1742    123              8                  16             yes      auto             1             6    fine_grained            basic           6       combined               0.0             None    Yes          True       False          False      10   [1.0, 1.0, 1.0]
# Epoch 10 | Time: 356.1s | Train Loss: 4.3558 | Train PPL: 77.93 | Val Loss: 4.7536 | Val PPL: 116.00 | Aux Loss: 0.2003    100              8                  16             yes      auto             1             6    fine_grained            basic           6       combined               0.0             None    Yes          True       False          False      10   [1.0, 1.0, 1.0]
# Epoch 10 | Time: 360.8s | Train Loss: 4.3536 | Train PPL: 77.76 | Val Loss: 4.7593 | Val PPL: 116.66 | Aux Loss: 0.1581    123              8                  16             yes      auto             3             6    fine_grained            basic           6       combined               0.0             None    Yes          True       False          False      10   [1.0, 1.0, 1.0]
# Epoch 10 | Time: 362.4s | Train Loss: 4.3512 | Train PPL: 77.57 | Val Loss: 4.7495 | Val PPL: 115.53 | Aux Loss: 0.1542    123              8                  16             yes      auto             1             6    fine_grained            basic           6       combined               0.0             None    Yes          True        True          False      10   [1.0, 1.0, 1.0]
# Epoch 10 | Time: 348.7s | Train Loss: 4.3574 | Train PPL: 78.05 | Val Loss: 4.7550 | Val PPL: 116.17 | Aux Loss: 0.1385    123              8                  16             yes      auto             1             6    fine_grained            basic           6       combined               0.0             None    Yes          True        True           True      10   [1.0, 1.0, 1.0]
# Epoch 10 | Time: 364.0s | Train Loss: 4.3712 | Train PPL: 79.14 | Val Loss: 4.7709 | Val PPL: 118.02 | Aux Loss: -0.0927   123              8                  16             yes      auto             1            11    fine_grained       basic_plus          11       combined               0.0             None    Yes          True        True           True      10   [1.0, 1.0, 1.0]
# Epoch 10 | Time: 443.8s | Train Loss: 4.3699 | Train PPL: 79.03 | Val Loss: 4.7688 | Val PPL: 117.78 | Aux Loss: -0.1117    20              8                  16             yes      auto             1            11    fine_grained       basic_plus          11       combined               0.0             None    Yes          True        True           True      10   [1.0, 1.0, 1.0]
# Epoch 10 | Time: 446.2s | Train Loss: 4.3616 | Train PPL: 78.38 | Val Loss: 4.7699 | Val PPL: 117.91 | Aux Loss: 0.2056     20              8                  16             yes      auto             1             6    fine_grained            basic           6       combined               0.0             None    Yes          True        True           True      10   [1.0, 1.0, 1.0]
# Epoch 11 | Time: 354.1s | Train Loss: 4.2616 | Train PPL: 70.92 | Val Loss: 4.7494 | Val PPL: 115.51 | Aux Loss: 0.2414    123              8                  16             yes      auto             1             6    fine_grained            basic           6       combined               0.0             None    Yes          True        True          False      15   [1.0, 1.0, 1.0]
# Epoch 11 | Time: 374.2s | Train Loss: 4.2616 | Train PPL: 70.92 | Val Loss: 4.7511 | Val PPL: 115.71 | Aux Loss: 0.3107    123              8                  16             yes      auto             1             6    fine_grained            basic           6       combined               0.0             None    Yes          True        True          False      15   [2.0, 3.0, 1.0]
# Epoch 11 | Time: 381.9s | Train Loss: 4.2609 | Train PPL: 70.87 | Val Loss: 4.7498 | Val PPL: 115.57 | Aux Loss: 0.3000    123              8                  16             yes      auto             1             6    fine_grained            basic           6       combined               0.0             None    Yes          True        True          False      15   [3.0, 5.0, 1.0]
