import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class LowRankLinear(nn.Module):
    def __init__(self, in_shape: int, out_shape: int, rank : int, base, bias : torch.Tensor):
        super().__init__()
        alpha_t = torch.zeros(out_shape, rank, requires_grad = True)
        beta_t = torch.zeros(rank, in_shape, requires_grad = True)
        self.alpha = nn.Parameter(alpha_t, requires_grad = True)
        self.beta = nn.Parameter(beta_t, requires_grad = True)
        self.bias = nn.Parameter(bias, requires_grad = True)
        self.base = base

    def forward(self, x):
        full_weight = torch.add(self.base, torch.matmul(self.alpha, self.beta))
        return F.linear(x, full_weight, self.bias)