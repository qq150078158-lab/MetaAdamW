"""A Self‑Attentive Meta‑Optimizer with Group‑Adaptive Learning Rates and Weight Decay"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.func import functional_call
from typing import Dict, List, Any, Callable, Tuple, Optional
# from loguru import logger

from .huw import MultiTaskLossWrapper


# 定义临时模型类，使用 functional_call 动态调用
class TempModel(nn.Module):
    def __init__(self, orig_model, param_dict):
        super().__init__()
        self.orig_model = orig_model
        self.param_dict = param_dict
    def forward(self, *args, **kwargs):
        return functional_call(self.orig_model, self.param_dict, args, kwargs)


# 基于分组自注意力机制的增强型 AdamW 优化器
class MetaAdamW(AdamW):
    """
    MetaAdamW: 基于分组自注意力机制的增强型 AdamW 优化器。

    该优化器对每个参数组（group）提取特征，然后通过一个轻量级 Transformer 编码器
    计算组间的注意力，为每个组输出个性化的学习率缩放因子 α 和权重衰减缩放因子 β。
    同时支持在线元学习（每 K 步）更新注意力模块，以提升泛化能力。

    主要特点：
        - 真正的 QKV 自注意力：捕捉参数组之间的依赖关系
        - 组内共享调制因子，降低计算开销
        - 可选的元学习更新（与 AdamW_AttnMeta 类似）
        - 通用接口：适配任意模型，自动根据优化器参数组构建组

    Args:
        params (iterable): 待优化参数或参数组。
        lr (float, optional): 全局学习率，默认 1e-3。
        betas (Tuple[float, float], optional): 动量系数，默认 (0.9, 0.999)。
        eps (float, optional): 分母稳定项，默认 1e-8。
        weight_decay (float, optional): 全局权重衰减，默认 1e-2。
        amsgrad (bool, optional): 是否使用 amsgrad（本实现忽略）。
        feature_dim (int, optional): 每个参数组的特征维度，默认 8。
        attn_hidden_dim (int, optional): 注意力模块隐藏层维度，默认 32。
        attn_layers (int, optional): Transformer 编码器层数，默认 1。
        attn_heads (int, optional): 注意力头数，默认 4。
        alpha_range (float, optional): 学习率缩放范围，最终 α ∈ [1-range/2, 1+range/2]，默认 1.0。
        beta_range (float, optional): 权重衰减缩放范围，默认 1.0。
        meta_update_freq (int, optional): 元学习更新频率（步数），默认 100。0 表示禁用。
        meta_lr (float, optional): 元学习的学习率，默认 1e-4。
        meta_weight_decay (float, optional): 注意力模块的 L2 正则化系数，默认 1e-4。
        total_steps (int, optional): 总训练步数，用于时间特征归一化。
        model(optional): 主模型实例，用于精细分组。
        group_strategy(str, optional): 分组策略，默认 'original'，可选 'fine_grained'
        feature_version(str, optional): 特征表达，默认 'basic'，
                                        可选 'basic', 'norm_basic', 'basic_plus', 'norm_basic_plus', 'enhanced'
                                        'basic_plus' 包含 4 个均值和 4 个标准差（共 8 个统计量）
                                        'norm_basic_plus' 会对这 8 个统计量进行跨组标准化
        include_time_step(bool, Optional): 特征中是否包含时间步，默认 True
        group_embed_dim(int, optional): 组嵌入向量的维度（仅在增强特征时使用），默认 4
        use_v_norms(bool, Optional): basic 相关特征中是否启用 二阶动量范数 特征，默认 False
        use_feature_gating(bool, Optional): 是否启用 特征门控 机制，默认 False
        feature_gating_sparsity(float, Optional): 启用特征门控时的正则化强度，默认 1e-4
        meta_objective(str, optional): 元学习目标，默认 'gradient'，可选 'loss_decrease', 'gen_gap', 'combined'
                                       分别对应：梯度方向，损失下降，泛化差距，三者混合
        val_loader(optional): 验证集数据加载器，用于计算验证损失，默认 None；
                              'loss_decrease' 或 'gen_gap' 或 'combined' 时必须提供
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False,
                 feature_dim=8, attn_hidden_dim=32, attn_layers=2, attn_heads=4,
                 alpha_range=1.0, beta_range=1.0,
                 meta_update_freq=100, meta_lr=1e-4,
                 meta_weight_decay=1e-4, total_steps=None,
                 model=None, group_strategy='original',
                 feature_version='basic', include_time_step=True, group_embed_dim=4,
                 use_v_norms=False,  # basic 相关特征中是否启用 二阶动量范数 特征
                 use_feature_gating=False, feature_gating_sparsity=1e-4,
                 meta_objective='gradient', val_loader=None,
                 # 固定权重列表 [w_grad, w_loss, w_gap]
                 aux_loss_fixed_weights: Optional[List[float]] = None,
                 # 是否启用同调不确定性加权
                 use_huw: bool = False,
                 task_types: List[str] = None,
                 # HUW 的业务优先级，默认 [1.0,1.0,1.0]
                 huw_priorities: Optional[List[float]] = None,
                 ):
        super().__init__(params, lr=lr, betas=betas, eps=eps,
                         weight_decay=weight_decay, amsgrad=False)

        self.feature_dim = feature_dim
        self.auto_feature_dim = (feature_dim is None)
        self.attn_hidden_dim = attn_hidden_dim
        self.attn_layers = attn_layers
        self.attn_heads = attn_heads
        self.alpha_range = alpha_range
        self.beta_range = beta_range
        self.meta_update_freq = meta_update_freq
        self.meta_lr = meta_lr
        self.meta_weight_decay = meta_weight_decay
        self.total_steps = total_steps
        self._step_count = 0

        self.model = model
        self.group_strategy = group_strategy

        self.feature_version = feature_version
        self.group_embed_dim = group_embed_dim
        self.include_time_step = include_time_step
        self.use_v_norms = use_v_norms
        self.use_feature_gating = use_feature_gating
        self.feature_gating_sparsity = feature_gating_sparsity
        self._feature_gates = None      # 将在构建注意力模块时初始化
        self.group_embeddings = None    # 将在构建组后初始化

        self.meta_objective = meta_objective
        self.val_loader = val_loader
        self._val_iter = None       # 用于迭代验证批次

        # 组信息：每个组包含的参数列表和对应的索引
        self.group_indices = None   # 每个组内参数在 all_params 中的索引
        self.all_params = None      # 所有参数的扁平列表
        self.param_to_group = None
        self.param_to_orig_group = None

        # 辅助损失权重
        self.aux_loss_fixed_weights = aux_loss_fixed_weights
        self.use_huw = use_huw
        self.task_types = task_types
        self.huw_priorities = huw_priorities if huw_priorities is not None else [1.0, 1.0, 1.0]
        self.huw = None

        # 注意力模块将在第一次 step 时构建
        self._attention_encoder = None
        self._output_proj = None

        # 保存用于加载状态的临时变量
        self._attention_encoder_state = None
        self._output_proj_state = None

    def _init_feature_dim(self):
        """根据特征版本和组嵌入维度计算所需的最小特征维度，并确保能被 attn_heads 整除。"""
        # 1. 计算基础维度
        if self.feature_version in ['basic', 'norm_basic']:
            # 基础特征：4均值 + 可选时间步 + 可选二阶动量范数
            base_dim = 4 + (1 if self.include_time_step else 0) + (1 if self.use_v_norms else 0)

        elif self.feature_version in ['basic_plus', 'norm_basic_plus']:
            # 基础特征plus：4均值 + 4标准差 + 可选时间步 + 可选二阶动量范数
            base_dim = 8 + (1 if self.include_time_step else 0) + (2 if self.use_v_norms else 0)

        else:  # 'enhanced'
            # 增强特征：9个统计量 + 可选时间步 + 组嵌入
            base_dim = 9 + (1 if self.include_time_step else 0) + self.group_embed_dim

        # 2. 自动模式：调整到能被 attn_heads 整除的最小合法维度
        if self.auto_feature_dim:
            # 找到 >= base_dim 且能被 self.attn_heads 整除的最小整数
            remainder = base_dim % self.attn_heads
            if remainder == 0:
                self.feature_dim = base_dim
            else:
                self.feature_dim = base_dim + (self.attn_heads - remainder)
            # logger.info(f"[MetaAdamW] Auto-set feature_dim to {self.feature_dim} "
            #             f"(base_dim={base_dim}, attn_heads={self.attn_heads})")
        else:
            # 用户指定模式：检查合法性，若不合法则调整并警告
            if self.feature_dim % self.attn_heads != 0:
                # 调整到最近的合法值（向上取整）
                remainder = self.feature_dim % self.attn_heads
                new_dim = self.feature_dim + (self.attn_heads - remainder)
                self.feature_dim = new_dim
                # logger.info(f"[MetaAdamW] WARNING: user-specified feature_dim={self.feature_dim} "
                #             f"not divisible by attn_heads={self.attn_heads}. Adjusted to {new_dim}.")

            # else:
            #     # 检查是否足够容纳基础特征
            #     if self.feature_dim < base_dim:
            #         logger.warning(f"[MetaAdamW] WARNING: user-specified feature_dim={self.feature_dim} "
            #                        f"is less than required minimum {base_dim}. Features will be truncated.")
            #     else:
            #         logger.info(f"[MetaAdamW] Using user-specified feature_dim={self.feature_dim} "
            #                     f"(minimum required {base_dim})")

    def _build_groups(self, params_list: List[torch.nn.Parameter]) -> None:
        """构建参数组，根据策略选择分组方式，并建立原始组映射"""
        self.all_params = params_list
        # 构建所有参数的 id 集合，用于快速判断
        param_ids = {id(p) for p in params_list}

        # 建立参数到原始参数组的映射
        self.param_to_orig_group = {}
        for orig_group_idx, group in enumerate(self.param_groups):
            for p in group['params']:
                if id(p) in param_ids:
                    self.param_to_orig_group[p] = orig_group_idx

        if self.group_strategy == 'original':
            self._build_groups_original(params_list)
        else:
            self._build_groups_fine_grained(params_list)

        # 使用增强特征
        if self.feature_version == 'enhanced':
            self._init_group_embeddings()

        # 分组构建完成后调用特征维度初始化
        self._init_feature_dim()

    def _build_groups_original(self, params_list: List[torch.nn.Parameter]) -> None:
        """
        根据优化器的 param_groups 自动构建分组。
        每个现有的 param_group 被视为一个独立的组。
        """
        self.all_params = params_list
        self.param_to_group = {}
        self.group_indices = []

        # 构建 id -> index 映射
        id_to_idx = {id(p): idx for idx, p in enumerate(params_list)}

        # 遍历所有参数组
        for group_idx, group in enumerate(self.param_groups):
            group_params = [p for p in group['params'] if id(p) in id_to_idx]
            if not group_params:
                continue
            indices = [id_to_idx[id(p)] for p in group_params]
            self.group_indices.append(indices)
            for p in group_params:
                self.param_to_group[p] = group_idx

    def _build_groups_fine_grained(self, params_list: List[torch.nn.Parameter]) -> None:
        """当前分组最多为 4种类型 × 3个深度桶 × 2种偏置 = 24组，在注意力模块中是可接受的。"""
        if self.model is None:
            raise ValueError("Fine-grained grouping requires model to be provided in optimizer constructor.")

        # 建立参数对象到名称的映射
        param_to_name = {id(p): name for name, p in self.model.named_parameters()}
        id_to_idx = {id(p): idx for idx, p in enumerate(params_list)}

        # 确定总层数（用于深度分桶）
        layer_indices = []
        for p in params_list:
            name = param_to_name.get(id(p))
            if name:
                # 提取数字索引（例如 "transformer.layers.0.self_attn" -> 0）
                import re
                match = re.search(r'\.(\d+)\.', name) or re.search(r'\.(\d+)$', name)
                if match:
                    layer_indices.append(int(match.group(1)))
        max_layer = max(layer_indices) if layer_indices else 0

        # 分组字典：组ID -> 参数索引列表
        groups = {}

        for p in params_list:
            idx = id_to_idx[id(p)]
            name = param_to_name.get(id(p))
            if name is None:
                # 无法获取名称，放入一个默认组
                group_id = ('unknown', 0, False)
            else:
                # 1. 确定层类型
                layer_type = self._infer_layer_type(name, p)
                # 2. 确定是否为偏置
                is_bias = (p.dim() == 1) or name.endswith('bias')
                # 3. 确定深度桶
                depth_bucket = self._get_depth_bucket(name, max_layer)
                group_id = (layer_type, depth_bucket, is_bias)

            groups.setdefault(group_id, []).append(idx)

        # 构建组列表和映射
        self.group_indices = []
        self.param_to_group = {}
        for group_idx, indices in enumerate(groups.values()):
            self.group_indices.append(indices)
            for idx in indices:
                p = params_list[idx]
                self.param_to_group[p] = group_idx

    def _get_module_by_name(self, name: str):
        """通过点号分隔的名称获取子模块（假设模型支持 get_submodule），在 PyTorch 1.9+ 中可用"""
        if self.model is None:
            return None
        try:
            return self.model.get_submodule(name)
        except AttributeError:
            return None

    def _infer_layer_type(self, name: str, param: torch.nn.Parameter) -> str:
        """根据参数名和参数本身推断层类型"""
        name_lower = name.lower()
        # 优先通过模块名判断
        if 'embed' in name_lower:
            return 'embedding'
        if 'attn' in name_lower or 'attention' in name_lower:
            return 'attention'
        if 'ffn' in name_lower or 'mlp' in name_lower or 'feed_forward' in name_lower:
            return 'ffn'
        if 'norm' in name_lower or 'layer_norm' in name_lower:
            return 'layernorm'
        # 对于线性层，如果无法通过名称识别，则尝试通过模块类型（需要获取模块对象）
        module = self._get_module_by_name('.'.join(name.split('.')[:-1])) if '.' in name else None
        if module is not None:
            if isinstance(module, nn.Linear):
                # 如果没有更细的分类，归为其他线性层
                return 'linear'
            if isinstance(module, nn.Embedding):
                return 'embedding'
            if isinstance(module, nn.MultiheadAttention):
                return 'attention'
            if isinstance(module, nn.LayerNorm):
                return 'layernorm'
        # 默认
        return 'other'

    def _get_depth_bucket(self, name: str, max_layer: int) -> int:
        """返回深度桶索引：0=浅层，1=中层，2=深层"""
        import re
        match = re.search(r'\.(\d+)\.', name) or re.search(r'\.(\d+)$', name)
        if not match:
            return 1  # 无层索引则归为中层
        layer = int(match.group(1))
        if max_layer <= 1:
            return 1
        if layer < max_layer / 3:
            return 0
        elif layer > 2 * max_layer / 3:
            return 2
        else:
            return 1

    def _init_group_embeddings(self):
        """这些嵌入是普通的 Parameter，但不由主优化器管理；它们的梯度将在元学习更新中手动处理。"""
        device = self.all_params[0].device if self.all_params else torch.device('cpu')
        self.group_embeddings = nn.ParameterList([
            nn.Parameter(torch.randn(self.group_embed_dim, device=device) * 0.1)
            for _ in range(len(self.group_indices))
        ])

    def _build_attention_module(self, num_groups: int, device) -> None:
        """根据组数构建 Transformer 编码器"""
        # 输入形状: (seq_len, batch, feature_dim) -> 这里 batch=1，seq_len=num_groups
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.feature_dim,
            nhead=self.attn_heads,
            dim_feedforward=self.attn_hidden_dim,
            dropout=0.0,
            batch_first=False,      # (seq, batch, feature)
            norm_first=True
        )
        self._attention_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.attn_layers
        ).to(dtype=torch.float32).to(device)

        # 输出维度 4: [alpha_raw, beta_raw, lambda1_raw, lambda2_raw]
        self._output_proj = nn.Linear(self.feature_dim, 4).to(dtype=torch.float32).to(device)

        # 初始化
        for p in self._attention_encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.xavier_uniform_(self._output_proj.weight)
        nn.init.zeros_(self._output_proj.bias)

        # 初始化 HUW（如果启用）
        if self.all_params:
            self._init_huw(self.all_params[0].device)

        # 初始化特征门控参数
        if self.use_feature_gating:
            # 为每个特征维度创建一个可学习的门控参数，初始化为 1
            self._feature_gates = nn.Parameter(
                torch.ones(self.feature_dim, device=device, dtype=torch.float32)
            )

        # 如果有加载的状态，则恢复
        if hasattr(self, '_attention_encoder_state') and self._attention_encoder_state is not None:
            self._attention_encoder.load_state_dict(self._attention_encoder_state)
            self._output_proj.load_state_dict(self._output_proj_state)
            self._attention_encoder_state = None
            self._output_proj_state = None

    def _get_param_name(self, p):
        """根据参数对象获取名称（需模型）"""
        if not hasattr(self, '_param_to_name'):
            if self.model is None:
                return ''
            self._param_to_name = {id(p): name for name, p in self.model.named_parameters()}
        return self._param_to_name.get(id(p), '')

    def _compute_max_layer(self):
        """从模型中提取最大层编号"""
        if not hasattr(self, '_max_layer'):
            max_layer = 0
            for name, _ in self.model.named_parameters():
                import re
                match = re.search(r'\.(\d+)\.', name) or re.search(r'\.(\d+)$', name)
                if match:
                    max_layer = max(max_layer, int(match.group(1)))
            self._max_layer = max_layer
        return self._max_layer

    def _get_layer_normalized(self, p):
        """归一化层编号"""
        name = self._get_param_name(p)
        if not name:
            return 0.0
        import re
        match = re.search(r'\.(\d+)\.', name) or re.search(r'\.(\d+)$', name)
        if not match:
            return 0.0
        layer = int(match.group(1))
        max_layer = self._compute_max_layer()
        if max_layer == 0:
            return 0.0
        return layer / max_layer

    def _extract_group_features(self, group_params: List[torch.nn.Parameter],
                                states: Dict, t_norm: torch.Tensor,
                                group_idx: int = -1) -> torch.Tensor:
        # norm_basic 也使用 basic 特征，标准化在 _compute_scaling_factors 中完成
        if self.feature_version in ['basic', 'norm_basic']:
            return self._extract_basic_features(group_params, states, t_norm)
        else:  # 'enhanced'
            return self._extract_enhanced_features(group_params, states, t_norm, group_idx)

    def _extract_basic_features(self, group_params: List[torch.nn.Parameter],
                                states: Dict, t_norm: torch.Tensor) -> torch.Tensor:
        """
        为一个参数组提取特征向量。

        Args:
            group_params: 组内参数列表
            states: 每个参数的状态字典（含 exp_avg, exp_avg_sq）
            t_norm: 归一化时间步 (scalar tensor)

        Returns:
            Tensor: 形状 (feature_dim,) 的特征向量
        """
        if len(group_params) == 0:
            # 如果组为空，返回零向量，长度取决于特征版本
            if self.feature_version in ['basic_plus', 'norm_basic_plus']:
                feat_len = 8 + (1 if self.include_time_step else 0) + (2 if self.use_v_norms else 0)
            else:
                feat_len = 4 + (1 if self.include_time_step else 0) + (1 if self.use_v_norms else 0)
            return torch.zeros(feat_len, dtype=torch.float32, device=t_norm.device)

        # 收集组内所有参数的统计量
        grad_norms = []
        m_norms = []
        p_norms = []
        cos_sims = []
        v_norms = []

        for p in group_params:
            state = states[p]
            grad = p.grad
            if grad is None:
                grad = torch.zeros_like(p)
            exp_avg = state['exp_avg']
            exp_avg_sq = state['exp_avg_sq']

            grad_norms.append(grad.norm().item())
            m_norms.append(exp_avg.norm().item())
            p_norms.append(p.data.norm().item())
            cos_sims.append(F.cosine_similarity(grad.view(-1), exp_avg.view(-1), dim=0).item())
            if self.use_v_norms:
                v_norms.append(exp_avg_sq.norm().item())

        # 转换为 numpy 数组方便计算
        grad_norms = np.array(grad_norms)
        m_norms = np.array(m_norms)
        p_norms = np.array(p_norms)
        cos_sims = np.array(cos_sims)
        if self.use_v_norms:
            v_norms = np.array(v_norms)

        # 计算均值和标准差（使用 ddof=0，即除以 N）
        mean_grad = grad_norms.mean()
        mean_m = m_norms.mean()
        mean_p = p_norms.mean()
        mean_cos = cos_sims.mean()
        if self.use_v_norms:
            mean_v = v_norms.mean()

        # 标准差：如果只有一个元素，std 为 0
        std_grad = grad_norms.std(ddof=0) if len(grad_norms) > 1 else 0.0
        std_m = m_norms.std(ddof=0) if len(m_norms) > 1 else 0.0
        std_p = p_norms.std(ddof=0) if len(p_norms) > 1 else 0.0
        std_cos = cos_sims.std(ddof=0) if len(cos_sims) > 1 else 0.0
        if self.use_v_norms:
            std_v = v_norms.std(ddof=0) if len(v_norms) > 1 else 0.0

        # 构建基础特征列表
        if self.feature_version in ['basic_plus', 'norm_basic_plus']:
            if self.use_v_norms:
                feat_list = [mean_grad, mean_m, mean_p, mean_cos, mean_v, std_grad, std_m, std_p, std_cos, std_v]
            else:
                feat_list = [mean_grad, mean_m, mean_p, mean_cos, std_grad, std_m, std_p, std_cos]
        else:  # 'basic' or 'norm_basic'
            if self.use_v_norms:
                feat_list = [mean_grad, mean_m, mean_p, mean_cos, mean_v]
            else:
                feat_list = [mean_grad, mean_m, mean_p, mean_cos]

        if self.include_time_step:
            feat_list.append(t_norm.item())

        group_feat = torch.tensor(feat_list, dtype=torch.float32, device=t_norm.device)

        # 若特征维度大于 len(feat_list)，则用零填充或扩展
        if self.feature_dim > len(feat_list):
            extra = torch.zeros(self.feature_dim - len(feat_list), dtype=torch.float32, device=t_norm.device)
            group_feat = torch.cat([group_feat, extra])

        return group_feat

    def _extract_enhanced_features(self, group_params, states, t_norm, group_idx):
        """
        增强特征包括：
            统计特征：梯度范数的均值与方差、动量范数的均值与方差、梯度稀疏度、动量稀疏度。
            结构特征：参数量对数、偏置比例、归一化层编号。
            时间步（与原相同）。
            可学习组嵌入（拼接在最后）。
        """
        grad_norms = []
        grad_norm_sq = []
        m_norms = []
        m_norm_sq = []
        grad_sparsity = []
        m_sparsity = []
        param_size_log = []
        is_bias = []
        layer_norm = []

        for p in group_params:
            state = states[p]
            grad = p.grad if p.grad is not None else torch.zeros_like(p)
            exp_avg = state['exp_avg']

            # 梯度范数
            g_norm = grad.norm().item()
            grad_norms.append(g_norm)
            grad_norm_sq.append(g_norm ** 2)
            # 动量范数
            m_norm = exp_avg.norm().item()
            m_norms.append(m_norm)
            m_norm_sq.append(m_norm ** 2)
            # 稀疏度（非零比例）
            g_density = (grad != 0).float().mean().item()
            grad_sparsity.append(1 - g_density)
            m_density = (exp_avg != 0).float().mean().item()
            m_sparsity.append(1 - m_density)
            # 参数量对数
            param_size_log.append(np.log(p.numel() + 1))
            # 偏置判断（根据名称或维度）
            is_bias_val = 0
            if self.model is not None:
                name = self._get_param_name(p)
                if name and (name.endswith('bias') or p.dim() == 1):
                    is_bias_val = 1.0
            is_bias.append(is_bias_val)
            # 归一化层编号
            layer_norm.append(self._get_layer_normalized(p))

        # 聚合统计量
        grad_mean = np.mean(grad_norms)
        grad_var = np.var(grad_norms) if len(grad_norms) > 1 else 0.0
        m_mean = np.mean(m_norms)
        m_var = np.var(m_norms) if len(m_norms) > 1 else 0.0
        grad_sparsity_mean = np.mean(grad_sparsity)
        m_sparsity_mean = np.mean(m_sparsity)
        param_size_mean = np.mean(param_size_log)
        is_bias_ratio = np.mean(is_bias)
        layer_norm_mean = np.mean(layer_norm)

        # 构建特征向量（9个基础统计量 + 时间步）
        stats = [
            grad_mean, grad_var, m_mean, m_var,
            grad_sparsity_mean, m_sparsity_mean,
            param_size_mean, is_bias_ratio, layer_norm_mean,
        ]
        if self.include_time_step:
            stats.append(t_norm.item())

        # 确保特征长度不超过 feature_dim - group_embed_dim
        stats_len = len(stats)
        available_len = self.feature_dim - self.group_embed_dim
        if stats_len > available_len:
            # 截断
            stats = stats[:available_len]
        else:
            # 补零
            pad = [0.0] * (available_len - stats_len)
            stats.extend(pad)
        group_feat = torch.tensor(stats, dtype=torch.float32, device=t_norm.device)

        # 拼接组嵌入
        if self.group_embeddings is not None and group_idx >= 0:
            embed = self.group_embeddings[group_idx]
            group_feat = torch.cat([group_feat, embed])
        else:
            # 如果没有组嵌入，补零
            pad = torch.zeros(self.group_embed_dim, dtype=torch.float32, device=t_norm.device)
            group_feat = torch.cat([group_feat, pad])

        return group_feat

    def _compute_scaling_factors(self, groups: List[List[int]], states: Dict,
                                  t_norm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        通过自注意力计算所有组的 α, β, λ1, λ2。

        Args:
            groups: 组索引列表，每个元素是该组参数在 all_params 中的索引
            states: 参数状态字典（映射到状态）
            t_norm: 归一化时间步 (scalar)

        Returns:
            alphas: Tensor 形状 (num_groups,)，每个组的 α
            betas:  Tensor 形状 (num_groups,)，每个组的 β
            lambdas1: Tensor (num_groups,) 平坦性项权重
            lambdas2: Tensor (num_groups,) 方向性项权重
        """
        if self._attention_encoder is None:
            self._build_attention_module(len(groups), t_norm.device)

        # 提取每个组的特征向量
        group_feats = []

        for group_idx, idxs in enumerate(groups):
            group_params = [self.all_params[i] for i in idxs]
            feat = self._extract_group_features(group_params, states, t_norm, group_idx)
            group_feats.append(feat)

        group_feats = torch.stack(group_feats, dim=0)   # (num_groups, feature_dim)

        # 特征门控
        if self.use_feature_gating and self._feature_gates is not None:
            # 门控权重通过 sigmoid 限制在 (0,1) 范围
            # 可选：如果希望直接使用参数值而不经过 sigmoid（允许负权重），可取消 sigmoid，但此时需要额外的正则化约束
            gates = torch.sigmoid(self._feature_gates)  # (feature_dim,)
            group_feats = group_feats * gates.unsqueeze(0)  # 逐元素相乘

        # 根据版本决定需要标准化的统计列数
        if self.feature_version == 'norm_basic':
            num_stats = 4 + (1 if self.use_v_norms else 0)
        elif self.feature_version == 'norm_basic_plus':
            num_stats = 8 + (2 if self.use_v_norms else 0)
        else:
            num_stats = 0  # 其他版本不进行标准化

        # 如果是 norm，对（统计量）进行标准化
        if num_stats > 0:
            stats_cols = group_feats[:, :num_stats]  # shape (num_groups, num_stats)
            mean = stats_cols.mean(dim=0, keepdim=True)
            std = stats_cols.std(dim=0, keepdim=True) + 1e-8
            stats_cols_norm = (stats_cols - mean) / std
            group_feats[:, :num_stats] = stats_cols_norm

        # 自注意力：输入形状 (seq_len, batch=1, feature_dim)
        group_feats = group_feats.unsqueeze(1)          # (num_groups, 1, feature_dim)
        attn_out = self._attention_encoder(group_feats) # (num_groups, 1, feature_dim)
        attn_out = attn_out.squeeze(1)                  # (num_groups, feature_dim)
        raw = self._output_proj(attn_out)               # (num_groups, 4)

        alpha_raw, beta_raw, lambda1_raw, lambda2_raw = raw[:, 0], raw[:, 1], raw[:, 2], raw[:, 3]
        alphas = 1.0 + self.alpha_range * (torch.sigmoid(alpha_raw) - 0.5)
        betas = 1.0 + self.beta_range * (torch.sigmoid(beta_raw) - 0.5)
        lambdas1 = torch.sigmoid(lambda1_raw)
        lambdas2 = torch.sigmoid(lambda2_raw)

        return alphas, betas, lambdas1, lambdas2

    def _get_t_norm(self, device: torch.device) -> torch.Tensor:
        """获取归一化时间步（标量张量）"""
        if self.total_steps is not None:
            t_norm = torch.tensor(self._step_count / self.total_steps,
                                  dtype=torch.float32, device=device)
        else:
            # 使用 sigmoid 归一化，确保输出为 float32
            t_val = self._step_count / 1000.0
            t_norm = torch.tensor(2 * torch.sigmoid(torch.tensor(t_val, dtype=torch.float32)).item() - 1,
                                  dtype=torch.float32, device=device)
        return t_norm

    def _get_val_batch(self, device):
        """返回一个验证批次（输入、目标）"""
        if self.val_loader is None:
            raise ValueError("val_loader is required for meta_objective 'loss_decrease', 'gen_gap', or 'combined'.")

        # 简单循环使用迭代器
        if self._val_iter is None:
            self._val_iter = iter(self.val_loader)
        try:
            batch = next(self._val_iter)
        except StopIteration:
            self._val_iter = iter(self.val_loader)
            batch = next(self._val_iter)
        inputs, targets = batch
        return inputs.to(device), targets.to(device)

    def set_val_loader(self, val_loader):
        """设置验证数据加载器，用于元学习目标需要验证集的情况。"""
        self.val_loader = val_loader
        self._val_iter = None  # 重置迭代器

    def _init_huw(self, device):
        if self.use_huw and self.huw is None:
            self.huw = MultiTaskLossWrapper(
                task_types=self.task_types,
                priorities=self.huw_priorities
            ).to(device)

    def _calc_original_grad(self, loss_fn, model, batch1):
        loss1 = loss_fn(model, batch1)
        grads1 = torch.autograd.grad(loss1, self.all_params, retain_graph=False, allow_unused=True)
        # 处理 None 梯度
        grads1 = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads1, self.all_params)]
        return grads1

    def _build_temp_param(self, alphas, betas):
        temp_params = []
        for param_idx, p in enumerate(self.all_params):
            # 获取原始组索引（用于优化器超参数）
            orig_group_idx = self.param_to_orig_group[p]
            # 获取精细组索引（用于缩放因子）
            fine_group_idx = self.param_to_group[p]

            alpha = alphas[fine_group_idx]
            beta = betas[fine_group_idx]

            state = self.state[p]
            if state.get('step', 0) == 0:
                # 若状态未初始化，则临时跳过（理论上 step>=1）
                temp_params.append(p.data)
                continue

            group = self.param_groups[orig_group_idx]
            beta1, beta2 = group['betas']
            eps = group['eps']
            lr = group['lr']
            wd = group['weight_decay']
            step = state['step']
            exp_avg = state['exp_avg']
            exp_avg_sq = state['exp_avg_sq']

            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step
            step_size = lr / bias_correction1
            denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(eps)
            update = exp_avg / denom
            if wd != 0:
                update.add_(p.data, alpha=(wd * beta).item())
            temp_p = p.data - step_size * alpha * update
            temp_params.append(temp_p)
        return temp_params

    def _calc_grads_temp(self, model, loss_fn, batch2, temp_params):
        # 构建参数名映射（模型参数顺序与 self.all_params 一致）
        param_names = [name for name, _ in model.named_parameters()]

        # 构建参数名到临时张量的映射
        temp_param_dict = {name: temp_p for name, temp_p in zip(param_names, temp_params)}

        # 创建临时模型实例
        temp_model = TempModel(model, temp_param_dict)

        # 使用提供的损失函数计算损失（内部会调用 temp_model 的前向传播）
        loss2 = loss_fn(temp_model, batch2)

        grads_temp = torch.autograd.grad(loss2, temp_params, retain_graph=True, allow_unused=True, create_graph=True)

        # 将可能的 None 替换为对应参数形状的零张量
        grads_temp = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads_temp, temp_params)]

        return grads_temp, temp_model

    def _calc_aux_loss_gradient(self, grads1, grads_temp, lambdas1, lambdas2):
        """梯度方向目标"""
        # 将梯度按组聚合
        group_g1 = []
        group_g2 = []
        for idxs in self.group_indices:
            g1_list = []
            g2_list = []
            for i in idxs:
                g1 = grads1[i].view(-1)
                g2 = grads_temp[i].view(-1)
                g1_list.append(g1)
                g2_list.append(g2)
            group_g1.append(torch.cat(g1_list))
            group_g2.append(torch.cat(g2_list))

        group_aux_losses = []
        for i, (g1, g2) in enumerate(zip(group_g1, group_g2)):
            norm_loss = g2.norm() ** 2
            cos_sim = F.cosine_similarity(g1, g2, dim=0)
            l1 = lambdas1[i]
            l2 = lambdas2[i]
            group_aux_loss = l1 * norm_loss - l2 * cos_sim
            group_aux_losses.append(group_aux_loss)

        aux_loss_gradient = torch.stack(group_aux_losses).mean()
        return aux_loss_gradient

    def _calc_aux_loss_loss_decrease(self, model, temp_model, loss_fn=None):
        """损失下降目标"""
        # 获取验证批次
        val_inputs, val_targets = self._get_val_batch(self.all_params[0].device)
        val_batch = (val_inputs, val_targets)

        # 当前模型在验证集上的损失
        with torch.no_grad():
            if loss_fn is not None:
                # 如果提供了通用的损失计算函数，直接使用
                loss_val_current = loss_fn(model, val_batch)
            else:
                _, loss_val_current = model(val_inputs, val_targets)

        # 临时模型在验证集上的损失
        if loss_fn is not None:
            # 如果提供了通用的损失计算函数，直接使用
            loss_val_temp = loss_fn(temp_model, val_batch)
        else:
            _, loss_val_temp = temp_model(val_inputs, val_targets)

        # 希望 loss_val_temp 小于 loss_val_current
        aux_loss_loss_decrease = loss_val_temp - loss_val_current

        return aux_loss_loss_decrease

    def _calc_aux_loss_gen_gap(self, model, batch1, loss_fn=None):
        """泛化差距目标"""
        # 验证损失（从验证集获取）
        val_inputs, val_targets = self._get_val_batch(self.all_params[0].device)
        val_batch = (val_inputs, val_targets)

        if loss_fn is not None:
            # 使用统一的 loss_fn 接口
            loss_train = loss_fn(model, batch1)
        else:
            # 训练损失（在 batch1 上）
            _, loss_train = model(batch1[0], batch1[1])

        with torch.no_grad():
            if loss_fn is not None:
                loss_val = loss_fn(model, val_batch)
            else:
                _, loss_val = model(val_inputs, val_targets)

        # 泛化差距：训练损失与验证损失之差的绝对值
        aux_loss_gen_gap = torch.abs(loss_train - loss_val)

        return aux_loss_gen_gap

    def _calc_aux_loss(self, grads1, grads_temp, lambdas1, lambdas2, batch1, model, temp_model, loss_fn=None):
        aux_loss = 0.0

        # 梯度方向目标
        if self.meta_objective in ['gradient', 'combined']:
            loss_gradient = self._calc_aux_loss_gradient(grads1, grads_temp, lambdas1, lambdas2)
            aux_loss = loss_gradient

        # 损失下降目标
        if self.meta_objective in ['loss_decrease', 'combined']:
            loss_loss_decrease = self._calc_aux_loss_loss_decrease(model, temp_model, loss_fn=loss_fn)
            aux_loss = loss_loss_decrease

        # 泛化差距目标
        if self.meta_objective in ['gen_gap', 'combined']:
            loss_gen_gap = self._calc_aux_loss_gen_gap(model, batch1, loss_fn=loss_fn)
            aux_loss = loss_gen_gap

        # 混合目标
        if self.meta_objective in ['combined']:
            if self.use_huw:
                # 使用同调不确定性加权
                aux_loss = self.huw([loss_gradient, loss_loss_decrease, loss_gen_gap])
            elif self.aux_loss_fixed_weights is not None:
                # 使用固定权重
                w1, w2, w3 = self.aux_loss_fixed_weights
                aux_loss = w1 * loss_gradient + w2 * loss_loss_decrease + w3 * loss_gen_gap
            else:
                # 默认简单相加
                aux_loss = loss_gradient + loss_loss_decrease + loss_gen_gap

        # 添加 L2 正则化
        mlp_reg = sum(p.norm(2) for p in self._attention_encoder.parameters()) + self._output_proj.weight.norm(2)
        aux_loss = aux_loss + self.meta_weight_decay * mlp_reg

        # 特征门控稀疏正则化
        if self.use_feature_gating and self._feature_gates is not None:
            # L1 正则化，使门控趋向于 0
            # 这里采用 L1 对原始参数进行正则化，并配合 sigmoid，实际效果是鼓励门控值向 0 或 1 两极分化
            l1_reg = self._feature_gates.abs().sum()
            aux_loss = aux_loss + self.feature_gating_sparsity * l1_reg

        return aux_loss

    def _update_param(self):
        with torch.no_grad():
            # 更新注意力模块参数
            for param in self._attention_encoder.parameters():
                if param.grad is not None:
                    param.data -= self.meta_lr * param.grad
                    param.grad = None
            if self._output_proj.weight.grad is not None:
                self._output_proj.weight.data -= self.meta_lr * self._output_proj.weight.grad
                self._output_proj.weight.grad = None
            if self._output_proj.bias.grad is not None:
                self._output_proj.bias.data -= self.meta_lr * self._output_proj.bias.grad
                self._output_proj.bias.grad = None

            # 更新组嵌入
            if hasattr(self, 'group_embeddings') and self.group_embeddings is not None:
                for embed in self.group_embeddings:
                    if embed.grad is not None:
                        embed.data -= self.meta_lr * embed.grad
                        embed.grad = None

            # 更新 HUW 参数
            if self.huw is not None:
                for param in self.huw.parameters():
                    if param.grad is not None:
                        param.data -= self.meta_lr * param.grad
                        param.grad = None

    @torch.no_grad()
    def step(self, closure=None):
        """
        执行单步参数更新（标准优化器接口）。
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step_count += 1

        # 确保参数扁平列表和分组已构建
        if self.all_params is None:
            all_params = []
            seen_ids = set()
            for group in self.param_groups:
                for p in group['params']:
                    if id(p) not in seen_ids:
                        seen_ids.add(id(p))
                        all_params.append(p)
            self._build_groups(all_params)

        # 获取当前时间步归一化值
        device = self.all_params[0].device
        t_norm = self._get_t_norm(device)

        # 预处理：收集所有参数的状态
        for p in self.all_params:
            state = self.state[p]
            if len(state) == 0:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

        # 计算各组缩放因子
        alphas, betas, _, _ = self._compute_scaling_factors(self.group_indices, self.state, t_norm)

        # 为每个参数应用更新
        param_idx = 0
        for group_idx, group in enumerate(self.param_groups):
            beta1, beta2 = group['betas']
            weight_decay = group['weight_decay']
            lr = group['lr']
            eps = group['eps']

            group_params = group['params']
            for p in group_params:
                if p.grad is None:
                    continue
                state = self.state[p]
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1

                # 更新动量
                exp_avg.mul_(beta1).add_(p.grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)

                # 偏差校正
                step = state['step']
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                step_size = lr / bias_correction1

                denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(eps)
                update = exp_avg / denom

                # 应用组调制因子
                fine_group_idx = self.param_to_group[p]
                alpha = alphas[fine_group_idx]
                beta = betas[fine_group_idx]

                if weight_decay != 0:
                    update.add_(p.data, alpha=(weight_decay * beta).item())
                p.data.add_(update, alpha=-step_size * alpha)

        return loss

    @torch.enable_grad()
    def update_attention(self, model: nn.Module, batch1: Any, batch2: Any,
                         loss_fn: Callable[[nn.Module, Any], torch.Tensor]) -> float:
        """
        执行注意力模块的元学习更新（与 AdamW_AttnMeta 类似）。

        Args:
            model (nn.Module): 主模型实例
            batch1: 第一个小批量数据
            batch2: 第二个小批量数据
            loss_fn (callable): 接受 (model, batch) 返回标量损失

        Returns:
            float: 辅助损失值
        """
        if self.meta_update_freq == 0: return 0.0

        # 确保分组信息已构建
        if self.all_params is None:
            all_params = []
            for group in self.param_groups:
                for p in group['params']:
                    if p not in all_params:
                        all_params.append(p)
            self._build_groups(all_params)

        # 保存原始参数数据
        orig_data = {p: p.data.clone() for p in self.all_params}

        # 获取当前时间步
        t_norm = self._get_t_norm(self.all_params[0].device)

        # 第一步：计算原始梯度 g (在 batch1 上)
        grads1 = self._calc_original_grad(loss_fn, model, batch1)

        # 第二步：计算当前组的缩放因子（包括 lambda1, lambda2）
        alphas, betas, lambdas1, lambdas2 = self._compute_scaling_factors(self.group_indices, self.state, t_norm)

        # 第三步：构建临时参数
        temp_params = self._build_temp_param(alphas, betas)

        # 第四步：计算临时参数在 batch2 上的梯度 g'
        grads_temp, temp_model = self._calc_grads_temp(model, loss_fn, batch2, temp_params)

        # 第五步：计算辅助损失
        aux_loss = self._calc_aux_loss(grads1, grads_temp, lambdas1, lambdas2, batch1, model, temp_model, loss_fn=loss_fn)

        # 第六步：反向传播，更新注意力模块，更新组嵌入
        aux_loss.backward()
        self._update_param()

        # 恢复原始参数
        for p, orig in orig_data.items(): p.data.copy_(orig)

        return aux_loss.item()

    def state_dict(self) -> Dict[str, Any]:
        """返回优化器状态字典，包含注意力模块参数"""
        state_dict = super().state_dict()
        if self._attention_encoder is not None:
            state_dict['attention_encoder'] = self._attention_encoder.state_dict()
            state_dict['output_proj'] = self._output_proj.state_dict()
        state_dict['_step_count'] = self._step_count
        state_dict['hyper_params'] = {
            'feature_dim': self.feature_dim,
            'attn_hidden_dim': self.attn_hidden_dim,
            'attn_layers': self.attn_layers,
            'attn_heads': self.attn_heads,
            'alpha_range': self.alpha_range,
            'beta_range': self.beta_range,
            'meta_update_freq': self.meta_update_freq,
            'meta_lr': self.meta_lr,
            'meta_weight_decay': self.meta_weight_decay,
            'total_steps': self.total_steps,
            'group_strategy': self.group_strategy,
            'feature_version': self.feature_version,
            'group_embed_dim': self.group_embed_dim,
            'use_v_norms': self.use_v_norms,
            'use_feature_gating': self.use_feature_gating,
            'feature_gating_sparsity': self.feature_gating_sparsity,
            'meta_objective': self.meta_objective,
            'include_time_step': self.include_time_step,
            'aux_loss_fixed_weights': self.aux_loss_fixed_weights,
            'use_huw': self.use_huw,
            'huw_priorities': self.huw_priorities,
        }
        if self.group_embeddings is not None:
            state_dict['group_embeddings'] = [embed.data.clone() for embed in self.group_embeddings]

        if self.huw is not None:
            state_dict['huw'] = self.huw.state_dict()

        if self._feature_gates is not None:
            state_dict['feature_gates'] = self._feature_gates.data

        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """加载优化器状态"""
        # 提取注意力模块的状态
        attn_encoder_sd = state_dict.pop('attention_encoder', None)
        output_proj_sd = state_dict.pop('output_proj', None)
        step_count = state_dict.pop('_step_count', 0)
        group_embeddings_list = state_dict.pop('group_embeddings', None)

        hyper_params = state_dict.pop('hyper_params', {})
        huw_sd = state_dict.pop('huw', None)
        feature_gates = state_dict.pop('feature_gates', None)

        self._step_count = step_count
        self.group_strategy = hyper_params.get('group_strategy', 'original')

        self.feature_version = hyper_params.get('feature_version', 'basic')
        self.group_embed_dim = hyper_params.get('group_embed_dim', 4)
        self.include_time_step = hyper_params.get('include_time_step', True)
        self.use_v_norms = hyper_params.get('use_v_norms', False)
        self.use_feature_gating = hyper_params.get('use_feature_gating', False)
        self.feature_gating_sparsity = hyper_params.get('feature_gating_sparsity', 1e-4)

        if group_embeddings_list is not None:
            self.group_embeddings = nn.ParameterList([nn.Parameter(embed) for embed in group_embeddings_list])

        self.meta_objective = hyper_params.get('meta_objective', 'gradient')

        self.aux_loss_fixed_weights = hyper_params.get('aux_loss_fixed_weights')
        self.use_huw = hyper_params.get('use_huw', False)
        self.huw_priorities = hyper_params.get('huw_priorities', [1.0, 1.0, 1.0])

        if self.use_huw and huw_sd is not None:
            self.huw = MultiTaskLossWrapper(num_tasks=3, priorities=self.huw_priorities)
            self.huw.load_state_dict(huw_sd)
        if self.use_feature_gating and feature_gates is not None:
            self._feature_gates = nn.Parameter(feature_gates)

        super().load_state_dict(state_dict)

        if attn_encoder_sd is not None and output_proj_sd is not None:
            # 存储这些状态，在第一次 step 时恢复
            self._attention_encoder_state = attn_encoder_sd
            self._output_proj_state = output_proj_sd
        # 恢复超参数
        for k, v in hyper_params.items():
            if hasattr(self, k):
                setattr(self, k, v)

