"""MetaAdamW 消融实验"""

import numpy as np
import torch


# ==================== 固定随机种子 ====================
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)


if __name__ == '__main__':
    # 消融实验：MiniGPT on WikiText-2 语言建模
    from experiments.ablation_wikitext2 import train_wikitext2
    train_wikitext2()

    # 消融实验：CIFAR-10 with ResNet-18 图像分类
    # from experiments.ablation_cifar10 import train_cifar10
    # train_cifar10()

    # 消融实验：IWSLT14 De-En with Transformer 机器翻译
    # from experiments.ablation_multi30k import train_multi30k
    # train_multi30k()

    # 消融实验：TSF with Transformer on ETTh1 时间序列预测
    # from experiments.ablation_etth1 import train_etth1
    # train_etth1()

    # 消融实验：LSTM on IMDB 情感分类
    # from experiments.ablation_imdb import train_imdb
    # train_imdb()

