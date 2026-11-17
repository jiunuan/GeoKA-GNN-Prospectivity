"""标准 B-spline KAN 线性层。"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class KANLinear(nn.Module):
    """使用固定网格和三次 B-spline 基函数的 KANLinear。"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        base_activation: str = "SiLU",
        grid_range: tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("KAN 输入和输出维度必须大于 0。")
        if grid_size <= 0 or spline_order <= 0:
            raise ValueError("grid_size 和 spline_order 必须大于 0。")
        if base_activation.lower() != "silu":
            raise ValueError("当前实验的 base_activation 必须为 SiLU。")
        if grid_range[0] >= grid_range[1]:
            raise ValueError("KAN grid_range 必须严格递增。")

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.grid_size = int(grid_size)
        self.spline_order = int(spline_order)
        self.base_activation_name = "SiLU"
        basis_count = self.grid_size + self.spline_order

        step = (grid_range[1] - grid_range[0]) / self.grid_size
        grid = torch.arange(
            -self.spline_order,
            self.grid_size + self.spline_order + 1,
            dtype=torch.float32,
        )
        grid = grid * step + grid_range[0]
        self.register_buffer(
            "grid",
            grid.unsqueeze(0).expand(self.in_features, -1).contiguous(),
        )
        self.base_activation = nn.SiLU()
        self.base_weight = nn.Parameter(
            torch.empty(self.out_features, self.in_features)
        )
        self.spline_weight = nn.Parameter(
            torch.empty(self.out_features, self.in_features, basis_count)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """初始化基础线性权重和 B-spline 系数。"""
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.normal_(self.spline_weight, mean=0.0, std=0.01)

    def b_splines(self, inputs: torch.Tensor) -> torch.Tensor:
        """计算输入最后一维上的 Cox-de Boor B-spline 基函数。"""
        flattened = inputs.reshape(-1, self.in_features)
        values = flattened.unsqueeze(-1)
        grid = self.grid
        bases = (
            (values >= grid[:, :-1]) & (values < grid[:, 1:])
        ).to(values.dtype)
        for order in range(1, self.spline_order + 1):
            left_numerator = values - grid[:, : -(order + 1)]
            left_denominator = (
                grid[:, order:-1] - grid[:, : -(order + 1)]
            ).clamp_min(1.0e-12)
            right_numerator = grid[:, order + 1 :] - values
            right_denominator = (
                grid[:, order + 1 :] - grid[:, 1:-order]
            ).clamp_min(1.0e-12)
            bases = (
                left_numerator / left_denominator * bases[..., :-1]
                + right_numerator / right_denominator * bases[..., 1:]
            )
        return bases.reshape(
            *inputs.shape[:-1],
            self.in_features,
            self.grid_size + self.spline_order,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """计算 SiLU 基础分支与 B-spline 分支之和。"""
        original_shape = inputs.shape[:-1]
        flattened = inputs.reshape(-1, self.in_features)
        base_output = F.linear(
            self.base_activation(flattened),
            self.base_weight,
        )
        spline_basis = self.b_splines(inputs).reshape(
            -1,
            self.in_features * (self.grid_size + self.spline_order),
        )
        spline_weight = self.spline_weight.reshape(
            self.out_features,
            -1,
        )
        spline_output = F.linear(spline_basis, spline_weight)
        return (base_output + spline_output).reshape(
            *original_shape,
            self.out_features,
        )


def spline_l1_penalty(module: nn.Module) -> torch.Tensor:
    """汇总模型中全部样条系数的平均 L1 惩罚。"""
    penalties = [
        child.spline_weight.abs().mean()
        for child in module.modules()
        if isinstance(child, KANLinear)
    ]
    if not penalties:
        return torch.zeros((), device=next(module.parameters()).device)
    return torch.stack(penalties).sum()
