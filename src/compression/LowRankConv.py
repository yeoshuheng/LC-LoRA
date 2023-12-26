import torch, math
import torch.nn as nn
import tensorly as tl
from tensorly.decomposition import parafac

def generate_rank(x, y):
    #return max(0, min(x, y) // 32)
    return min(min(x, y), 8)

class LowRankConv(nn.Module):
    def __init__(self, in_shape: int, out_shape: int,
                 base, bias : torch.Tensor, scaling : int = -1, rank : int = -1):
        """
        @param in_shape, out_shape : Layer dimensions as per nn.Linear
        @param rank : Rank of the decomposition. 
            (if rank is -1, we use 'min(in_shape, out_shape)//2' as our rank instead.)
        @param base : Initial base weight of the layer (W), kept frozen during training.
        @param bias : Initial bias of the layer, trainable.
        @param scaling : Scaling factor of the LoRA decomposition.
        """
        super().__init__()
        if rank == -1:
            rank = generate_rank(in_shape, out_shape)
        if scaling == -1:
            scaling = 0.5
        self.scaling = scaling
        self.base = base.clone()
        self.base.requires_grad = False
        self.bias = bias
        a, b, _, d = base.shape
        self.alpha = nn.Parameter(
            torch.empty((rank, b, 1, 1), dtype = torch.float32, requires_grad = True))
        self.beta = nn.Parameter(
            torch.empty((rank, 1, d, d), dtype = torch.float32, requires_grad = True))
        self.gamma = nn.Parameter(
            torch.empty((a, rank, 1, 1), dtype = torch.float32, requires_grad = True))
        torch.nn.init.kaiming_uniform_(self.alpha, a =  math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.beta, a =  math.sqrt(5))
        torch.nn.init.zeros_(self.gamma)

    def forward(self, x):
        x = self.alpha(x)
        x = self.beta(x)
        x = self.gamma(x)
        x = self.eta(x)
        return self.base + self.scaling * x + self.bias