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
        l, f, v, h = parafac(base.data, rank = rank, init = "svd")
        self.alpha = nn.Conv2d(in_channels = f.shape[0], out_channels = f.shape[1], 
                                     kernel_size=1, stride=1, padding=0, bias = False)
        self.beta = nn.Conv2d(in_channels = v.shape[0], out_channels = v.shape[1], 
                                     kernel_size=1, stride=1, padding=0, bias = False)
        self.gamma = nn.Conv2d(in_channels = h.shape[0], out_channels = h.shape[1],
                               kernel_size=1, stride=1, padding=0, bias = False)
        self.eta = nn.Conv2d(in_channels = l.shape[0], out_channels = l.shape[1],
                             kernel_size=1, stride=1, padding=0, bias = False)
        torch.nn.init.kaiming_uniform_(self.alpha, a =  math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.beta, a =  math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.gamma, a =  math.sqrt(5))
        torch.nn.init.zeros_(self.eta)

    def forward(self, x):
        x = self.alpha(x)
        x = self.beta(x)
        x = self.gamma(x)
        x = self.eta(x)
        return self.base + self.scaling * x + self.bias