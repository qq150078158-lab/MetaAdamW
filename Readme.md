# MetaAdamW: A Self-Attentive Meta-Optimizer with Group-Adaptive Learning Rates and Weight Decay

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the implementation of **MetaAdamW**, a novel optimizer that integrates a self‑attention mechanism and meta‑learning into the AdamW framework. MetaAdamW dynamically modulates per‑group learning rates and weight decay by extracting statistical features from parameter groups and using a lightweight Transformer encoder. A meta‑learning objective – combining gradient alignment, loss decrease, and generalization gap – is balanced via **a key novel priority‑injected homoscedastic uncertainty weighting (HUW)**, enabling domain‑specific guidance for automatic loss balancing.

The code supports extensive ablation studies on five diverse tasks:
- Time series forecasting (ETTh1) with Transformer encoder
- Language modeling (WikiText-2) with MiniGPT
- Machine translation (Multi30k De→En) with Transformer seq2seq
- Image classification (CIFAR-10) with ResNet-18
- Sentiment analysis (IMDB) with bidirectional LSTM

For details, please refer to our paper:  
**"A Self‑Attentive Meta‑Optimizer with Group‑Adaptive Learning Rates and Weight Decay"** (to appear).  
<!-- After publication, add BibTeX citation here. -->

---

## Key Features

- **Group‑adaptive modulation** – per‑group learning rate (α) and weight decay (β) via self‑attention.
- **Meta‑learning of the attention module** – periodic updates using a composite objective (gradient alignment, loss decrease, generalization gap).
- **Priority‑injected HUW** – extends homoscedastic uncertainty weighting with task‑specific priorities that scale the regularization terms.
- **Multiple feature versions** – `basic`, `basic_plus`, `enhanced`, and their normalized variants.
- **Flexible grouping strategies** – original PyTorch parameter groups or fine‑grained grouping (layer type, depth bucket, bias).
- **Comprehensive ablation support** – all configurations are controlled via a central `config` dictionary in each experiment script.

---

## Experiments & Ablations

Each task has its own configuration file (`ablation_*.py`) with a detailed `config` dictionary. Key tunable hyperparameters:

| Parameter | Description |
|-----------|-------------|
| `optimizer_type` | `'AdamW'` or `'MetaAdamW'` |
| `meta_update_freq` | Steps between meta‑updates |
| `attn_layers`, `attn_hidden_dim`, `attn_heads` | Transformer encoder size |
| `feature_version` | `'basic'`, `'basic_plus'`, `'enhanced'`, etc. |
| `group_strategy` | `'original'` or `'fine_grained'` |
| `meta_objective` | `'gradient'`, `'loss_decrease'`, `'gen_gap'`, `'combined'` |
| `use_huw` + `huw_priorities` | Enable priority‑injected uncertainty weighting |

All experiments log CSV files to `./logs/` and save the best model checkpoint. Early stopping (patience 2) is applied by default.

---

## Results Summary

MetaAdamW consistently outperforms standard AdamW across five tasks:

| Task | Metric | AdamW | MetaAdamW | Improvement | Time Overhead |
|------|--------|-------|-----------|-------------|---------------|
| ETTh1 (forecast) | MSE | 0.006147 | **0.005885** | **+4.26%** | **-7.20%**    |
| WikiText-2 (LM) | PPL | 120.47 | **115.51** | **+4.12%** | **-17.11%**   |
| Multi30k (MT) | PPL | 2.0297 | **1.9690** | **+2.99%** | **+27.35%**   |
| CIFAR-10 (classif.) | Acc | 87.02% | **88.05%** | **+1.18%** | **+27.58%**   |
| IMDB (sentiment) | Acc | 74.53% | **82.79%** | **+11.08%** | **+172.53%**  |

Training time changes vary: reductions of up to 17.11% (WikiText-2) or increases up to +172.5% (IMDB) depending on task and meta‑update frequency.

Detailed ablation results and optimal hyperparameters per task are provided in the paper (Appendix).

---

## Citation

If you use MetaAdamW in your research, please cite:

```bibtex
@article{zhao2026metaadamw,
  title={A Self-Attentive Meta-Optimizer with Group-Adaptive Learning Rates and Weight Decay},
  author={JiangBo Zhao, ZhaoXin Liu},
  journal={},
  year={2026}
}

