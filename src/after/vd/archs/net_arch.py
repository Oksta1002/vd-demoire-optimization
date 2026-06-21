import torch
import torch.nn as nn
from einops import rearrange
from vd.utils.registry import ARCH_REGISTRY
from vd.archs.arch_util import Downsample, Upsample
from vd.archs.mdnet_arch import MoireDecouplingNetwork
from vd.archs.tanet_arch import TemporalAlignmentNetwork


class EncodingNetwork(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super(EncodingNetwork, self).__init__()
        self.feat_encode1 = Downsample(in_channels, out_channels // 2, activation=True)
        self.feat_encode2 = Downsample(out_channels // 2, out_channels, activation=True)

    def forward(self, x):
        # (B*T, 3, H, W) -> (B*T, C, H/4, W/4)
        feat = self.feat_encode1(x)  # (B*T, C/2, H/2, W/2)
        feat = self.feat_encode2(feat)  # (B*T, C, H/4, W/4)
        return feat


class DecodingNetwork(nn.Module):
    def __init__(self, in_channels=64, out_channels=3):
        super(DecodingNetwork, self).__init__()
        self.feat_decode1 = Upsample(in_channels, in_channels // 2, activation=True)
        self.feat_decode2 = Upsample(in_channels // 2, out_channels)

    def forward(self, x):
        # (B, C, H/4, W/4) -> (B, 3, H, W)
        feat = self.feat_decode1(x)   # (B, C/4, H/2, W/2)
        feat = self.feat_decode2(feat) # (B, 3, H, W)
        return feat


@ARCH_REGISTRY.register()
class MTNet(nn.Module):
    """
    MTNet: Moire-Decoupling and Temporal-Alignment Network for Video Demoireing.

    Args:
        num_feat (int): Number of feature channels. Default: 64.
        input_frames (int): Number of input frames. Default: 3.
        dw_expansion_factor (float): DW expansion factor in MDNet. Default: 2.
        ffn_expansion_factor (float): FFN expansion factor in MDNet. Default: 2.
        num_blocks (list[int]): MDB blocks per MDNet stage. Default: [1, 1, 1].

    Returns:
        Tuple[Tensor, Tensor, Tensor]: (aligned_output, clean_center, moire_center), each (B, 3, H, W).
    """
    def __init__(
        self,
        num_feat=64,
        input_frames=3,
        dw_expansion_factor=2,
        ffn_expansion_factor=2,
        num_blocks=[1, 1, 1]
    ):
        super().__init__()
        self.input_frames = input_frames

        self.enc = EncodingNetwork(in_channels=3, out_channels=num_feat)
        self.MDNet = MoireDecouplingNetwork(
            num_feat=num_feat,
            dw_expansion_factor=dw_expansion_factor,
            ffn_expansion_factor=ffn_expansion_factor,
            num_blocks=num_blocks
        )
        self.TANet = TemporalAlignmentNetwork(num_feat=num_feat, input_frames=input_frames)
        self.dec_a = DecodingNetwork(in_channels=num_feat, out_channels=3)
        self.dec_c = DecodingNetwork(in_channels=num_feat, out_channels=3)
        self.dec_m = DecodingNetwork(in_channels=num_feat, out_channels=3)

    def extract_center_frame(self, x):
        """(B*T, C, H, W) -> center frame (B, C, H, W)"""
        x = rearrange(x, '(b t) c h w -> b t c h w', b=x.size(0) // self.input_frames, t=self.input_frames)
        return x[:, self.input_frames // 2].contiguous()

    def forward(self, x):
        feat_d = rearrange(x, 'b t c h w -> (b t) c h w')
        feat_d = self.enc(feat_d)
        feat_c, feat_m = self.MDNet(feat_d)
        feat_a = self.TANet(feat_c)
        feat_a = self.dec_a(feat_a)
        feat_c = self.dec_c(feat_c)
        feat_c = rearrange(feat_c, '(b t) c h w -> b t c h w', b=feat_c.size(0)//self.input_frames, t=self.input_frames)
        feat_m = self.dec_m(feat_m)
        feat_m = rearrange(feat_m, '(b t) c h w -> b t c h w', b=feat_m.size(0)//self.input_frames, t=self.input_frames)
        return feat_a, feat_c, feat_m
