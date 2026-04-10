"""Multi30k De-En with Transformer 机器翻译消融实验"""

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

    # Multi30k De-En with Transformer 机器翻译特定配置
    'dataset': 'Multi30k',           # Multi30k De-En
    'max_len': 64,                  # 最大序列长度（截断）
    'batch_size': 64,
    'vocab_size': 10000,            # 源/目标共享词汇表大小
    'min_freq': 2,                  # 最小词频

    # Multi30k De-En with Transformer 模型参数
    'd_model': 256,
    'nhead': 8,
    'num_encoder_layers': 4,
    'num_decoder_layers': 4,
    'dim_feedforward': 1024,
    'dropout': 0.2,

    # 优化器通用设置
    'lr': 5e-4,
    'weight_decay': 0.01,

    # MetaAdamW 专属设置
    'meta_update_freq': 454,            # 元更新频率（步数），454 steps/epoch
    'feature_dim': 6,                   # 每个参数组的特征维度（在 'enhanced' 时需设置 16 以确保足够容纳）；若为 None 则自动计算有效特征维度
    'attn_hidden_dim': 64,              # 注意力MLP隐藏维度 16
    'attn_layers': 128,                   # 注意力层数 2
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
    'aux_loss_fixed_weights': [1.0, 0.3, 0.3],   # 固定权重，若为 None 则使用简单相加 (combined_weights=None)
    'use_huw': True,                             # 是否启用同调不确定性加权（优先级高于固定权重）
    'task_types': ['regression', 'regression', 'regression'],
    'huw_priorities': [2.0, 5.0, 1.0],           # HUW 业务优先级，默认等权重

    # 训练设置
    'epochs': 25,
    'log_interval': 10,
    # 一般3轮时主模型大致可以收敛到较平坦区域，然后再启用注意力模块，这样可以避免早期的不稳定影响后续元学习
    'warmup_epoch': 1,                           # 在主模型训练到 第几轮 时，优化器才启动元学习

    # 早停设置
    'patience': 2,
    'min_delta': 1e-3,
    'restore_best_weights': True,
}


# 定义损失函数闭包（用于元学习）
def translation_loss_fn(model, batch):
    """模型接受 (src, tgt) 返回损失"""
    src, tgt = batch
    logits = model(src, tgt[:, :-1])  # 教师强制，输入目标序列去掉最后一个 token
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1), ignore_index=0)
    return loss


# ==================== Transformer 机器翻译模型 ====================
class PositionalEncoding(nn.Module):
    """位置编码模块"""
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (seq_len, batch, d_model)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerTranslator(nn.Module):
    """标准的 Transformer 序列到序列模型"""
    def __init__(self, vocab_size, d_model=256, nhead=8,
                 num_encoder_layers=4, num_decoder_layers=4,
                 dim_feedforward=1024, dropout=0.1, max_len=64):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,      # 默认 (seq, batch, feature)
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        """
        src: (batch, src_len)
        tgt: (batch, tgt_len)
        输出: (batch, tgt_len, vocab_size)
        """
        # 嵌入并添加位置编码
        src = self.embedding(src) * math.sqrt(self.d_model)  # (batch, src_len, d_model)
        src = src.transpose(0, 1)                            # (src_len, batch, d_model)
        src = self.pos_encoder(src)

        tgt = self.embedding(tgt) * math.sqrt(self.d_model)  # (batch, tgt_len, d_model)
        tgt = tgt.transpose(0, 1)                            # (tgt_len, batch, d_model)
        tgt = self.pos_encoder(tgt)

        # 生成目标掩码（因果掩码）
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(0)).to(src.device)

        # 前向传播
        output = self.transformer(src, tgt, tgt_mask=tgt_mask)  # (tgt_len, batch, d_model)
        output = output.transpose(0, 1)                         # (batch, tgt_len, d_model)
        logits = self.fc_out(output)                            # (batch, tgt_len, vocab_size)
        return logits


# ==================== 数据集准备 ====================
def load_multi30k():
    """使用 torchtext 内置的 Multi30k 数据集（德-英翻译）"""
    try:
        from torchtext.datasets import Multi30k
        from torchtext.data.utils import get_tokenizer
        from torchtext.vocab import build_vocab_from_iterator
    except ImportError:
        raise ImportError("需要安装 torchtext: pip install torchtext")

    # 加载 Multi30k 数据集（自动下载，约 30MB）
    # 注意：torchtext 0.12+ 返回的是迭代器，需要 list() 转换
    train_iter, val_iter, test_iter = Multi30k(
        root='./data',
        language_pair=('de', 'en'),
        split=('train', 'valid', 'test')
    )

    train_list = list(train_iter)
    val_list = list(val_iter)

    # 分词器（使用 spacy，失败则回退到空格分词）
    try:
        tokenizer = get_tokenizer('spacy', language='de_core_news_sm')
    except Exception:
        tokenizer = lambda x: x.split()

    # 构建共享词汇表
    def yield_tokens(data_iter, language):
        for src, tgt in data_iter:
            if language == 'src':
                yield tokenizer(src)
            else:
                yield tokenizer(tgt)

    vocab = build_vocab_from_iterator(
        yield_tokens(train_list, 'src'),
        specials=['<pad>', '<unk>', '<bos>', '<eos>'],
        max_tokens=config['vocab_size'],
        min_freq=config['min_freq']
    )
    vocab.set_default_index(vocab['<unk>'])

    return train_list, val_list, vocab, tokenizer


def encode_sentence(sentence, vocab, tokenizer, max_len):
    """将句子编码为 token ids，添加 <bos> 和 <eos>"""
    tokens = tokenizer(sentence)
    ids = [vocab['<bos>']] + [vocab[token] for token in tokens] + [vocab['<eos>']]
    if len(ids) > max_len:
        ids = ids[:max_len-1] + [vocab['<eos>']]  # 保留 <eos>
    return ids


def collate_batch(batch, vocab, max_len):
    """批处理函数，对源和目标序列进行填充"""
    src_batch, tgt_batch = [], []
    pad_idx = vocab['<pad>']
    for src, tgt in batch:
        src_ids = encode_sentence(src, vocab, tokenizer, max_len)
        tgt_ids = encode_sentence(tgt, vocab, tokenizer, max_len)
        src_batch.append(torch.tensor(src_ids, dtype=torch.long))
        tgt_batch.append(torch.tensor(tgt_ids, dtype=torch.long))

    # 填充
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=pad_idx)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=pad_idx)
    return src_batch, tgt_batch


class TranslationDataset(Dataset):
    def __init__(self, data_list, vocab, tokenizer, max_len):
        self.data = data_list
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src, tgt = self.data[idx]
        return src, tgt


def get_dataloaders(batch_size, max_len, vocab_size, min_freq):
    global tokenizer  # 在外部声明，以便在 collate 中使用
    train_list, val_list, vocab, tokenizer = load_multi30k()
    config['vocab_size'] = len(vocab)  # 更新实际词汇表大小
    logger.info(f"词汇表大小: {len(vocab)}")

    train_dataset = TranslationDataset(train_list, vocab, tokenizer, max_len)
    val_dataset = TranslationDataset(val_list, vocab, tokenizer, max_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_batch(batch, vocab, max_len)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_batch(batch, vocab, max_len)
    )
    return train_loader, val_loader


# ==================== 训练函数 ====================
def train_epoch(model, device, train_loader, optimizer, epoch,
                hyperadamw=False, meta_update_freq=None):
    """
    训练一个 epoch，返回 平均损失, 平均辅助损失 等
    """
    model.train()
    total_loss = 0.0
    total_tokens = 0
    aux_loss_total = 0.0
    aux_loss_count = 0
    aux_loss = 0.0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch}', leave=True, ncols=140)

    if hyperadamw and meta_update_freq is not None and meta_update_freq > 0 and epoch >= config.get('warmup_epoch', 1):
        # 创建两个迭代器用于元学习
        train_iter1 = iter(train_loader)
        train_iter2 = iter(train_loader)

    for batch_idx, (src, tgt) in enumerate(pbar):
        src, tgt = src.to(device), tgt.to(device)

        optimizer.zero_grad()
        # 教师强制：输入目标序列去掉最后一个 token
        logits = model(src, tgt[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1), ignore_index=0)
        loss.backward()
        optimizer.step()

        # 统计
        batch_tokens = (tgt[:, 1:] != 0).sum().item()
        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens

        # 元学习更新
        if hyperadamw and meta_update_freq and meta_update_freq > 0 and epoch >= config.get('warmup_epoch', 1):
            if (batch_idx + 1) % meta_update_freq == 0:
                try:
                    src1, tgt1 = next(train_iter1)
                except StopIteration:
                    train_iter1 = iter(train_loader)
                    src1, tgt1 = next(train_iter1)
                try:
                    src2, tgt2 = next(train_iter2)
                except StopIteration:
                    train_iter2 = iter(train_loader)
                    src2, tgt2 = next(train_iter2)

                src1, tgt1 = src1.to(device), tgt1.to(device)
                src2, tgt2 = src2.to(device), tgt2.to(device)

                aux_loss = optimizer.update_attention(
                    model,
                    (src1, tgt1),
                    (src2, tgt2),
                    translation_loss_fn
                )
                aux_loss_total += aux_loss
                aux_loss_count += 1

        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'ppl': f'{math.exp(loss.item()):.4f}',
            'aux_loss': f'{aux_loss:.4f}'
        })

    avg_loss = total_loss / total_tokens
    avg_ppl = math.exp(avg_loss)
    avg_aux_loss = aux_loss_total / aux_loss_count if aux_loss_count > 0 else 0.0
    return avg_loss, avg_ppl, avg_aux_loss


def validate(model, device, val_loader):
    """
    验证，返回 (平均损失, 困惑度)
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for src, tgt in val_loader:
            src, tgt = src.to(device), tgt.to(device)
            logits = model(src, tgt[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1), ignore_index=0)
            batch_tokens = (tgt[:, 1:] != 0).sum().item()
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens
    avg_loss = total_loss / total_tokens
    avg_ppl = math.exp(avg_loss)
    return avg_loss, avg_ppl


# ==================== 主程序 ====================
def train_multi30k():
    # 配置优化器
    optimizer_type = config['optimizer_type']

    # 日志
    device = torch.device(config['device'])
    logger.info(f"Device: {device}")
    logger.info(f"Optimizer: {optimizer_type}")

    # 加载数据
    train_loader, val_loader = get_dataloaders(
        batch_size=config['batch_size'],
        max_len=config['max_len'],
        vocab_size=config['vocab_size'],
        min_freq=config['min_freq']
    )

    # 创建模型
    model = TransformerTranslator(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_encoder_layers=config['num_encoder_layers'],
        num_decoder_layers=config['num_decoder_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout'],
        max_len=config['max_len']
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
    log_file = os.path.join(log_dir, f'{optimizer_type}_multi30k_seed{config["seed"]}.csv').replace('\\', '/')
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
                    f'Train Loss: {train_loss:.4f} | Train PPL: {train_ppl:.4f} | '
                    f'Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.4f} | '
                    f'Aux Loss: {train_aux_loss:.4f}')

        # 保存最佳模型
        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            torch.save(model.state_dict(), f'./logs/best_model_{optimizer_type}_multi30k.pt')

        # 写入日志
        with open(log_file, 'a') as f:
            f.write(f'{epoch},{train_loss:.4f},{train_ppl:.4f},{val_loss:.4f},{val_ppl:.4f},'
                    f'{train_aux_loss:.4f},{epoch_time:.2f}\n')

        # 早停检查
        if early_stopping(val_loss, model):
            logger.warning(f"Early stopping triggered at epoch {epoch}")
            break

    logger.success(f'Success, log file: {log_file}')


# ==================== 实验日志 ====================

# AdamW 基线 (patience=2, epochs=50)
# -------------------------------------------------------------------------------------------------------------------------
# Epoch  1 | Time: 953.2s | Train Loss: 1.3028 | Train PPL: 3.6797 | Val Loss: 0.9956 | Val PPL: 2.7065 | Aux Loss: 0.0000
# Epoch  2 | Time: 936.6s | Train Loss: 0.9691 | Train PPL: 2.6355 | Val Loss: 0.9404 | Val PPL: 2.5611 | Aux Loss: 0.0000
# Epoch  3 | Time: 916.1s | Train Loss: 0.9178 | Train PPL: 2.5039 | Val Loss: 0.8654 | Val PPL: 2.3760 | Aux Loss: 0.0000
# Epoch  4 | Time: 906.9s | Train Loss: 0.8850 | Train PPL: 2.4231 | Val Loss: 0.8524 | Val PPL: 2.3454 | Aux Loss: 0.0000
# Epoch  5 | Time: 903.5s | Train Loss: 0.8620 | Train PPL: 2.3680 | Val Loss: 0.8186 | Val PPL: 2.2673 | Aux Loss: 0.0000
# Epoch  6 | Time: 898.2s | Train Loss: 0.8395 | Train PPL: 2.3153 | Val Loss: 0.8256 | Val PPL: 2.2833 | Aux Loss: 0.0000
# Epoch  7 | Time: 908.0s | Train Loss: 0.8253 | Train PPL: 2.2827 | Val Loss: 0.7850 | Val PPL: 2.1924 | Aux Loss: 0.0000
# Epoch  8 | Time: 891.5s | Train Loss: 0.8082 | Train PPL: 2.2439 | Val Loss: 0.7625 | Val PPL: 2.1436 | Aux Loss: 0.0000
# Epoch  9 | Time: 932.7s | Train Loss: 0.7964 | Train PPL: 2.2176 | Val Loss: 0.7562 | Val PPL: 2.1302 | Aux Loss: 0.0000
# Epoch 10 | Time: 893.1s | Train Loss: 0.7843 | Train PPL: 2.1908 | Val Loss: 0.7434 | Val PPL: 2.1032 | Aux Loss: 0.0000
# Epoch 11 | Time: 873.8s | Train Loss: 0.7719 | Train PPL: 2.1639 | Val Loss: 0.7423 | Val PPL: 2.1008 | Aux Loss: 0.0000
# Epoch 12 | Time: 877.8s | Train Loss: 0.7618 | Train PPL: 2.1421 | Val Loss: 0.7390 | Val PPL: 2.0938 | Aux Loss: 0.0000
# Epoch 13 | Time: 881.3s | Train Loss: 0.7516 | Train PPL: 2.1203 | Val Loss: 0.7327 | Val PPL: 2.0806 | Aux Loss: 0.0000
# Epoch 14 | Time: 879.3s | Train Loss: 0.7442 | Train PPL: 2.1048 | Val Loss: 0.7390 | Val PPL: 2.0938 | Aux Loss: 0.0000
# Epoch 15 | Time: 882.8s | Train Loss: 0.7363 | Train PPL: 2.0882 | Val Loss: 0.7304 | Val PPL: 2.0758 | Aux Loss: 0.0000
# Epoch 16 | Time: 886.5s | Train Loss: 0.7309 | Train PPL: 2.0769 | Val Loss: 0.7342 | Val PPL: 2.0838 | Aux Loss: 0.0000
# Epoch 17 | Time: 869.5s | Train Loss: 0.7208 | Train PPL: 2.0560 | Val Loss: 0.7148 | Val PPL: 2.0438 | Aux Loss: 0.0000
# Epoch 18 | Time: 853.8s | Train Loss: 0.7172 | Train PPL: 2.0486 | Val Loss: 0.7118 | Val PPL: 2.0376 | Aux Loss: 0.0000
# Epoch 19 | Time: 876.1s | Train Loss: 0.7088 | Train PPL: 2.0315 | Val Loss: 0.7079 | Val PPL: 2.0297 | Aux Loss: 0.0000
# Epoch 20 | Time: 883.5s | Train Loss: 0.7034 | Train PPL: 2.0206 | Val Loss: 0.7135 | Val PPL: 2.0412 | Aux Loss: 0.0000
# Epoch 21 | Time: 896.0s | Train Loss: 0.6981 | Train PPL: 2.0100 | Val Loss: 0.7097 | Val PPL: 2.0334 | Aux Loss: 0.0000
# Early stopping triggered at epoch 21

# MetaAdamW 最优 (patience=2, epochs=30)
# -------------------------------------------------------------------------------------------------------------------------

# Epoch  1 | Time: 997.6s | Train Loss: 1.3962 | Train PPL: 4.0399 | Val Loss: 0.9857 | Val PPL: 2.6796 | Aux Loss: 0.2405
# Epoch  2 | Time: 999.1s | Train Loss: 0.9705 | Train PPL: 2.6393 | Val Loss: 0.9138 | Val PPL: 2.4939 | Aux Loss: 0.2598
# Epoch  3 | Time: 974.5s | Train Loss: 0.9224 | Train PPL: 2.5154 | Val Loss: 0.8767 | Val PPL: 2.4029 | Aux Loss: 0.2450
# Epoch  4 | Time: 979.2s | Train Loss: 0.8869 | Train PPL: 2.4275 | Val Loss: 0.8440 | Val PPL: 2.3257 | Aux Loss: 0.2656
# Epoch  5 | Time: 971.8s | Train Loss: 0.8620 | Train PPL: 2.3678 | Val Loss: 0.8229 | Val PPL: 2.2772 | Aux Loss: 0.2696
# Epoch  6 | Time: 965.2s | Train Loss: 0.8413 | Train PPL: 2.3194 | Val Loss: 0.8065 | Val PPL: 2.2401 | Aux Loss: 0.2665
# Epoch  7 | Time: 970.0s | Train Loss: 0.8237 | Train PPL: 2.2789 | Val Loss: 0.7840 | Val PPL: 2.1902 | Aux Loss: 0.2531
# Epoch  8 | Time: 962.7s | Train Loss: 0.8081 | Train PPL: 2.2436 | Val Loss: 0.7719 | Val PPL: 2.1638 | Aux Loss: 0.2422
# Epoch  9 | Time: 961.3s | Train Loss: 0.7914 | Train PPL: 2.2066 | Val Loss: 0.7547 | Val PPL: 2.1269 | Aux Loss: 0.3008
# Epoch 10 | Time: 959.9s | Train Loss: 0.7792 | Train PPL: 2.1798 | Val Loss: 0.7449 | Val PPL: 2.1062 | Aux Loss: 0.2468
# Epoch 11 | Time: 957.4s | Train Loss: 0.7661 | Train PPL: 2.1513 | Val Loss: 0.7640 | Val PPL: 2.1469 | Aux Loss: 0.3208
# Epoch 12 | Time: 950.7s | Train Loss: 0.7554 | Train PPL: 2.1286 | Val Loss: 0.7242 | Val PPL: 2.0630 | Aux Loss: 0.2686
# Epoch 13 | Time: 958.4s | Train Loss: 0.7470 | Train PPL: 2.1107 | Val Loss: 0.7247 | Val PPL: 2.0641 | Aux Loss: 0.2860
# Epoch 14 | Time: 949.5s | Train Loss: 0.7361 | Train PPL: 2.0878 | Val Loss: 0.7174 | Val PPL: 2.0491 | Aux Loss: 0.2878
# Epoch 15 | Time: 953.1s | Train Loss: 0.7275 | Train PPL: 2.0700 | Val Loss: 0.7215 | Val PPL: 2.0575 | Aux Loss: 0.2988
# Epoch 16 | Time: 952.6s | Train Loss: 0.7198 | Train PPL: 2.0541 | Val Loss: 0.7128 | Val PPL: 2.0398 | Aux Loss: 0.2929
# Epoch 17 | Time: 946.7s | Train Loss: 0.7113 | Train PPL: 2.0366 | Val Loss: 0.7054 | Val PPL: 2.0246 | Aux Loss: 0.2773
# Epoch 18 | Time: 938.8s | Train Loss: 0.7048 | Train PPL: 2.0235 | Val Loss: 0.7082 | Val PPL: 2.0303 | Aux Loss: 0.2535
# Epoch 19 | Time: 944.3s | Train Loss: 0.6973 | Train PPL: 2.0084 | Val Loss: 0.7039 | Val PPL: 2.0216 | Aux Loss: 0.2366
# Epoch 20 | Time: 937.7s | Train Loss: 0.6897 | Train PPL: 1.9932 | Val Loss: 0.6855 | Val PPL: 1.9848 | Aux Loss: 0.2359
# Epoch 21 | Time: 942.6s | Train Loss: 0.6818 | Train PPL: 1.9774 | Val Loss: 0.6905 | Val PPL: 1.9946 | Aux Loss: 0.2466
# Epoch 22 | Time: 936.6s | Train Loss: 0.6768 | Train PPL: 1.9675 | Val Loss: 0.6819 | Val PPL: 1.9777 | Aux Loss: 0.2637
# Epoch 23 | Time: 947.0s | Train Loss: 0.6718 | Train PPL: 1.9577 | Val Loss: 0.6877 | Val PPL: 1.9891 | Aux Loss: 0.2656
# Epoch 24 | Time: 943.5s | Train Loss: 0.6649 | Train PPL: 1.9443 | Val Loss: 0.6790 | Val PPL: 1.9719 | Aux Loss: 0.2464
# Epoch 25 | Time: 941.6s | Train Loss: 0.6575 | Train PPL: 1.9299 | Val Loss: 0.6775 | Val PPL: 1.9690 | Aux Loss: 0.3317
# Success

# MetaAdamW 实验终点                                                                                               meta_update_freq    attn_layers     attn_hidden_dim     total_steps     λ1,λ2  warmup_epoch    attn_heads  group_strategy  feature_version feature_dim meta_objective   encoder_dropout combined_weights    HUW inc_time_step use_v_norms feature_gating  epochs    huw_priorities
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Epoch 15 | Time: 945.9s | Train Loss: 0.7313 | Train PPL: 2.0778 | Val Loss: 0.7119 | Val PPL: 2.0378 | Aux Loss: 0.0369    454              8                  16             yes      auto             1             6    fine_grained            basic           6       combined               0.0             None    Yes          True        True          False      30   [1.0, 3.0, 5.0]
# Epoch  4 | Time: 1067.7s | Train Loss: 0.9908 | Train PPL: 2.6933 | Val Loss: 0.9624 | Val PPL: 2.6180 | Aux Loss: 0.1533    45             64                 128             yes      auto             1            11    fine_grained       basic_plus          11       combined               0.0             None    Yes          True        True           True      25   [1.0, 3.0, 5.0]
# Epoch 15 | Time: 1055.8s | Train Loss: 0.7363 | Train PPL: 2.0882 | Val Loss: 0.7099 | Val PPL: 2.0338 | Aux Loss: -0.0814   45              8                  16             yes      auto             1             6    fine_grained            basic           6       combined               0.0             None    Yes          True        True          False      25   [1.0, 3.0, 5.0]
# Epoch  9 | Time: 951.0s | Train Loss: 0.7987 | Train PPL: 2.2227 | Val Loss: 0.7516 | Val PPL: 2.1205 | Aux Loss: 0.0558    454              8                  16             yes      auto             1             6    fine_grained            basic           6       combined               0.0             None    Yes          True        True          False      25   [1.0, 1.0, 1.0]
# Epoch 14 | Time: 952.0s | Train Loss: 0.7435 | Train PPL: 2.1034 | Val Loss: 0.7146 | Val PPL: 2.0435 | Aux Loss: 0.0495    454              8                  16             yes      auto             1             6    fine_grained            basic           6       combined               0.0             None    Yes          True        True          False      25   [1.0, 5.0, 3.0]
# Epoch 18 | Time: 927.6s | Train Loss: 0.7126 | Train PPL: 2.0394 | Val Loss: 0.7080 | Val PPL: 2.0300 | Aux Loss: -0.0945   454             64                  64             yes      auto             1             6    fine_grained            basic           6       combined               0.0             None    Yes          True        True          False      25   [1.0, 3.0, 5.0]
# Epoch 17 | Time: 991.8s | Train Loss: 0.7113 | Train PPL: 2.0366 | Val Loss: 0.7054 | Val PPL: 2.0246 | Aux Loss: 0.2809    454            128                  64             yes      auto             1             6    fine_grained            basic           6       combined               0.0             None    Yes          True        True          False      25   [2.0, 4.0, 1.0]
# Epoch 16 | Time: 1009.4s | Train Loss: 0.7216 | Train PPL: 2.0577 | Val Loss: 0.7105 | Val PPL: 2.0350 | Aux Loss: 0.2734   454            128                  64             yes      auto             1             6    fine_grained            basic           6       combined               0.0             None    Yes          True        True          False      30   [2.0, 5.0, 1.0]
# Epoch 25 | Time: 941.6s | Train Loss: 0.6575 | Train PPL: 1.9299 | Val Loss: 0.6775 | Val PPL: 1.9690 | Aux Loss: 0.3317    454            128                  64             yes      auto             1             6    fine_grained            basic           6       combined               0.0             None    Yes          True        True          False      25   [2.0, 5.0, 1.0]

