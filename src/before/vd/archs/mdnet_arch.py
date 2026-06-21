import torch
import torch.nn as nn
import torch.nn.functional as F
from vd.archs.arch_util import Downsample, Upsample, LayerNorm2d


class SimpleGate(nn.Module):
    """Split channels in half and apply element-wise multiplication."""
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class FeedForward(nn.Module):
    """Depth-wise conv FFN with GELU gating."""
    def __init__(self, dim, expansion_factor):
        super(FeedForward, self).__init__()
        hidden_dim = int(dim * expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_dim, 1)
        self.dw_conv = nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim)
        self.project_out = nn.Conv2d(hidden_dim // 2, dim, 1)

    def forward(self, x):
        feat = self.project_in(x)
        feat1, feat2 = self.dw_conv(feat).chunk(2, dim=1)
        feat = F.gelu(feat1) * feat2
        return self.project_out(feat)


class SimpleFeedForward(nn.Module):
    """1x1 conv FFN with SimpleGate."""
    def __init__(self, dim, expansion_factor):
        super(SimpleFeedForward, self).__init__()
        self.project_in = nn.Conv2d(dim, dim * expansion_factor, 1)
        self.SG = SimpleGate()
        self.project_out = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        feat = self.project_in(x)
        feat = self.SG(feat)
        return self.project_out(feat)


class SpatialExplicitChannelSplit(nn.Module):
    """
    Dual-Branch Spatial Moire Estimator (DSME).

    Applies multi-scale dilated depthwise convolutions (d=1,4,7) to capture
    moire patterns at different frequencies, then splits into clean/moire branches
    via complementary channel attention (SCA).

    Args:
        dim (int): Number of input channels.
    """
    def __init__(self, dim):
        super().__init__()
        self.clean_pw1 = nn.Conv2d(dim, dim, 1)
        self.moire_pw1 = nn.Conv2d(dim, dim, 1)

        # Multi-scale dilated DW conv to capture moire at different frequencies
        self.score_dw_d1 = nn.Conv2d(dim, dim, 3, padding=1,  stride=1, groups=dim, dilation=1)
        self.score_dw_d4 = nn.Conv2d(dim, dim, 3, padding=4,  stride=1, groups=dim, dilation=4)
        self.score_dw_d7 = nn.Conv2d(dim, dim, 3, padding=7,  stride=1, groups=dim, dilation=7)

        self.clean_SCA = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, 1),
        )
        self.moire_SCA = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, 1),
        )

        self.clean_pw2 = nn.Conv2d(dim, dim // 2, 1)
        self.moire_pw2 = nn.Conv2d(dim, dim // 2, 1)

    def forward(self, x):
        # Shared spatial activation: aggregate multi-scale moire responses
        activated_feat = self.score_dw_d1(x) + self.score_dw_d4(x) + self.score_dw_d7(x)

        c = F.gelu(self.clean_pw1(x)) * activated_feat
        m = F.gelu(self.moire_pw1(x)) * activated_feat
        c = c * self.clean_SCA(c)
        m = m * self.moire_SCA(m)
        return self.clean_pw2(c), self.moire_pw2(m)


class MDB(nn.Module):
    """
    Moire Decoupling Block (MDB).

    Expands features via DW conv, splits into clean/moire branches via DSME,
    then refines each branch with LayerNorm + FFN and learnable residual scaling.

    Args:
        dim (int): Number of input channels.
        dw_expansion_factor (float): Expansion factor for depth-wise operations.
        ffn_expansion_factor (float): Expansion factor for feed-forward networks.
        drop_out_rate (float): Dropout rate. Default: 0.
    """
    def __init__(self, dim, dw_expansion_factor=2, ffn_expansion_factor=2, drop_out_rate=0.):
        super().__init__()
        dw_dim = dim * dw_expansion_factor

        self.init_norm = LayerNorm2d(dim)
        self.init_pw = nn.Conv2d(dim, dw_dim, 1)
        self.init_dw = nn.Conv2d(dw_dim, dw_dim, 3, padding=1, groups=dw_dim)
        self.split = SpatialExplicitChannelSplit(dim=dw_dim)

        self.clean_dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.clean_beta = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        self.clean_norm = LayerNorm2d(dim)
        self.clean_ffn = FeedForward(dim, ffn_expansion_factor)
        self.clean_dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.clean_gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)

        self.moire_dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.moire_beta = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        self.moire_norm = LayerNorm2d(dim)
        self.moire_ffn = FeedForward(dim, ffn_expansion_factor)
        self.moire_dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.moire_gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)

    def forward(self, inp, moire_inp):
        x = self.init_norm(inp)
        x = self.init_dw(self.init_pw(x))

        # Split expanded features into clean/moire via DSME
        x, m = self.split(x)

        moire = self.moire_beta * self.moire_dropout1(m) + moire_inp
        m = self.moire_dropout2(self.moire_ffn(self.moire_norm(moire)))

        clean = self.clean_beta * self.clean_dropout1(x) + inp
        x = self.clean_dropout2(self.clean_ffn(self.clean_norm(clean)))

        return clean + x * self.clean_gamma, moire + m * self.moire_gamma


class MoireDecouplingNetwork(nn.Module):
    """
    Moire Decoupling Network (MDNet).

    3-level U-Net where each level runs parallel clean/moire branches through MDB blocks.
    Skip connections concatenate encoder and decoder features at each level.

    Args:
        num_feat (int): Base number of feature channels. Default: 64.
        dw_expansion_factor (float): Expansion factor for depth-wise operations. Default: 2.
        ffn_expansion_factor (float): Expansion factor for feed-forward networks. Default: 2.
        num_blocks (list[int]): Number of MDB blocks at each stage [l1, l2, l3]. Default: [4, 6, 8].
    """
    def __init__(self, num_feat=64, dw_expansion_factor=2, ffn_expansion_factor=2, num_blocks=[4, 6, 8]):
        super(MoireDecouplingNetwork, self).__init__()

        def make_blocks(dim, n):
            return nn.ModuleList([
                MDB(dim=dim, dw_expansion_factor=dw_expansion_factor, ffn_expansion_factor=ffn_expansion_factor)
                for _ in range(n)
            ])

        self.clean_init = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.moire_init = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # Encoder
        self.encoder_level1 = make_blocks(num_feat, num_blocks[0])
        self.down1 = Downsample(num_feat, num_feat * 2)
        self.moire_down1 = Downsample(num_feat, num_feat * 2)
        self.encoder_level2 = make_blocks(num_feat * 2, num_blocks[1])
        self.down2 = Downsample(num_feat * 2, num_feat * 4)
        self.moire_down2 = Downsample(num_feat * 2, num_feat * 4)
        self.encoder_level3 = make_blocks(num_feat * 4, num_blocks[2])

        # Decoder
        self.up2 = Upsample(num_feat * 4, num_feat * 2)
        self.moire_up2 = Upsample(num_feat * 4, num_feat * 2)
        self.concat_conv2 = nn.Conv2d(num_feat * 4, num_feat * 2, 1)
        self.moire_concat_conv2 = nn.Conv2d(num_feat * 4, num_feat * 2, 1)
        self.decoder_level2 = make_blocks(num_feat * 2, num_blocks[1])
        self.up1 = Upsample(num_feat * 2, num_feat)
        self.moire_up1 = Upsample(num_feat * 2, num_feat)
        self.concat_conv1 = nn.Conv2d(num_feat * 2, num_feat, 1)
        self.moire_concat_conv1 = nn.Conv2d(num_feat * 2, num_feat, 1)
        self.decoder_level1 = make_blocks(num_feat, num_blocks[0])

        self.refine = make_blocks(num_feat, num_blocks[0])

        self.clean_last = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.moire_last = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

    def forward(self, x):
        clean_l1 = self.clean_init(x)
        moire_l1 = self.moire_init(x)
        for block in self.encoder_level1:
            clean_l1, moire_l1 = block(clean_l1, moire_l1)

        clean_l2 = self.down1(clean_l1)
        moire_l2 = self.moire_down1(moire_l1)
        for block in self.encoder_level2:
            clean_l2, moire_l2 = block(clean_l2, moire_l2)

        clean_l3 = self.down2(clean_l2)
        moire_l3 = self.moire_down2(moire_l2)
        for block in self.encoder_level3:
            clean_l3, moire_l3 = block(clean_l3, moire_l3)

        # Decode level 2: upsample + skip concat
        clean_l2 = self.concat_conv2(torch.cat([self.up2(clean_l3), clean_l2], 1))
        moire_l2 = self.moire_concat_conv2(torch.cat([self.moire_up2(moire_l3), moire_l2], 1))
        for block in self.decoder_level2:
            clean_l2, moire_l2 = block(clean_l2, moire_l2)

        # Decode level 1: upsample + skip concat
        clean_l1 = self.concat_conv1(torch.cat([self.up1(clean_l2), clean_l1], 1))
        moire_l1 = self.moire_concat_conv1(torch.cat([self.moire_up1(moire_l2), moire_l1], 1))
        for block in self.decoder_level1:
            clean_l1, moire_l1 = block(clean_l1, moire_l1)

        for block in self.refine:
            clean_l1, moire_l1 = block(clean_l1, moire_l1)

        return self.clean_last(clean_l1) + x, self.moire_last(moire_l1) + x
