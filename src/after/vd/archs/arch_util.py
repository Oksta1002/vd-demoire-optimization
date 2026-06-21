import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from .dcn import ModulatedDeformConvPack, modulated_deform_conv


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/weight_init.py
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            'mean is more than 2 std from [a, b] in nn.init.trunc_normal_. '
            'The distribution of values may be incorrect.',
            stacklevel=2)

    with torch.no_grad():
        low = norm_cdf((a - mean) / std)
        up = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * low - 1, 2 * up - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class DCNv2Pack(ModulatedDeformConvPack):
    def forward(self, x, feat):
        out = self.conv_offset(feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        offset_absmean = torch.mean(torch.abs(offset))
        if offset_absmean > 50:
            print(f'Offset abs mean is {offset_absmean}, larger than 50.')
        return torchvision.ops.deform_conv2d(
            x, offset, self.weight, self.bias,
            self.stride, self.padding, self.dilation, mask
        )


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, activation=False):
        super().__init__()
        if activation:
            self.body = nn.Sequential(
                nn.Conv2d(in_channels, out_channels // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.PixelUnshuffle(2)
            )
        else:
            self.body = nn.Sequential(
                nn.Conv2d(in_channels, out_channels // 4, 3, 1, 1),
                nn.PixelUnshuffle(2)
            )

    def forward(self, x):
        return self.body(x)
    

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, activation=False):
        super().__init__()
        if activation:
            self.body = nn.Sequential(
                nn.PixelShuffle(2),
                nn.Conv2d(in_channels // 4, out_channels, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True)
            )
        else:
            self.body = nn.Sequential(
                nn.PixelShuffle(2),
                nn.Conv2d(in_channels // 4, out_channels, 3, 1, 1)  
            )

    def forward(self, x):
        return self.body(x)


class LayerNormFunction(torch.autograd.Function):
    """Custom CUDA-friendly layer norm operating on (B, C, H, W) tensors."""
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class LayerNorm1d(nn.Module):
    def __init__(self, dim):
        super(LayerNorm1d, self).__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.normalized_shape = dim

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class ImplicitChannelSort(nn.Module):
    """
    Implicit Channel Sorting based on Moire Estimation.

    Uses combined GAP+GMP statistics through an MLP to generate per-channel
    sorting scores as a soft proxy for moire degree — no explicit FFT needed.

    Args:
        dim (int): Number of input channels.
        reduction (int): Reduction factor for MLP hidden dimension. Default: 4.
    """
    def __init__(self, dim, reduction=4):
        super().__init__()
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.score_net = nn.Sequential(
            nn.Linear(dim * 2, dim // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        avg_out = self.avg_pool(x).view(B, C)
        max_out = self.max_pool(x).view(B, C)
        scores = self.score_net(torch.cat([avg_out, max_out], dim=1))
        _, sort_idx = torch.sort(scores, dim=1)
        sort_idx_expanded = sort_idx.view(B, C, 1, 1).expand(-1, -1, H, W)
        return torch.gather(x, dim=1, index=sort_idx_expanded)
