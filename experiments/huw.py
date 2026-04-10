"""同调不确定性加权"""
import torch
import torch.nn as nn
from typing import List, Union, Dict, Optional, Iterable


# 通用版：同调不确定性加权 + 业务优先级（正确作用于正则项）
class MultiTaskLossWrapper(nn.Module):
    """
    同调不确定性加权 (Homoscedastic Uncertainty Weighting) + 业务优先级（正确作用于正则项）

    论文: "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics"
          (Kendall et al., CVPR 2018)

    参数:
        task_types (List[str]): 每个任务的类型，可选 'classification' 或 'regression'。
        priorities (List[float], optional): 每个任务的优先级（作用于正则项），默认均为 1.0。
        init_log_var (float, optional): 初始 log(σ²) 值，默认为 0.0 (即 σ=1)。
        device (torch.device, optional): 内部参数存储的设备，默认为 使用默认设备。

    用法:
        # 初始化
        loss_wrapper = MultiTaskLossWrapper(
            task_types=['classification', 'regression', 'regression'],
            priorities=[2.0, 1.0, 1.0],
            init_log_var=0.0,
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )

        # 在训练循环中
        loss_type = ...   # 分类损失 (Cross-entropy)
        loss_qty = ...    # 回归损失 (MSE)
        loss_lev = ...    # 回归损失 (MSE)
        total_loss = loss_wrapper([loss_type, loss_qty, loss_lev])
        # 或者使用字典:
        total_loss = loss_wrapper({'type': loss_type, 'qty': loss_qty, 'lev': loss_lev})
    """

    def __init__(
        self,
        task_types: List[str],
        priorities: Union[List[float], None] = None,
        init_log_var: float = 0.0,
        device: Optional[torch.device] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ):
        super().__init__()
        self.num_tasks = len(task_types)
        self.task_types = task_types

        # 检查任务类型合法性
        valid_types = {'classification', 'regression'}
        for t in task_types:
            if t not in valid_types:
                raise ValueError(f"任务类型 '{t}' 不支持，必须是 {valid_types}")

        # 初始化优先级（默认全1）
        if priorities is None:
            priorities = [1.0] * self.num_tasks
        else:
            if len(priorities) != self.num_tasks:
                raise ValueError("priorities 长度必须与 task_types 一致")

        # 注册为缓冲区，以便随模型移动
        self.register_buffer('priorities', torch.tensor(priorities, dtype=torch.float32, device=device))

        # 可学习的 s=log(σ²) 参数
        self.log_vars = nn.Parameter(torch.full((self.num_tasks,), init_log_var, device=device))

    def forward(self, losses: Union[List[torch.Tensor], Dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        参数:
            losses: 可以是与任务顺序对应的损失列表，或字典（键为任务名，值对应损失）。
                    若为字典，内部将按初始化顺序匹配任务类型。
        返回:
            总损失 (torch.Tensor)
        """
        # 将输入统一为列表形式（按任务顺序）
        if isinstance(losses, dict):
            # 假设任务顺序就是初始化顺序，直接取 values 会丢失顺序，故需按 task_types 顺序取
            loss_list = [losses.get(k, None) for k in self.task_types]
            if any(l is None for l in loss_list):
                raise KeyError(f"字典缺少任务 {self.task_types[loss_list.index(None)]}")
            losses = loss_list

        elif not isinstance(losses, Iterable):
            raise TypeError("losses 应为列表或字典")

        if len(losses) != self.num_tasks:
            raise ValueError(f"损失数量 {len(losses)} 与任务数量 {self.num_tasks} 不符")

        total_loss = 0.0
        for i, (loss_val, task_type, priority, log_var) in enumerate(
            zip(losses, self.task_types, self.priorities, self.log_vars)
        ):
            # 损失标量，需保持计算图
            if not isinstance(loss_val, torch.Tensor):
                raise TypeError(f"任务 {i} 的损失必须是 torch.Tensor")

            precision = torch.exp(-log_var)          # 1/σ²
            if task_type == 'regression':
                # 回归: 0.5 * exp(-s) * L + 0.5 * s * priority
                term = 0.5 * precision * loss_val + 0.5 * log_var * priority

            elif task_type == 'classification':
                # 分类: exp(-s) * L + s * priority
                term = precision * loss_val + log_var * priority

            else:
                raise ValueError(f"未知任务类型: {task_type}")

            total_loss = total_loss + term

        return total_loss

