"""
Temporal Alignment Network (TANet) — EAMamba SSM implementation.

- FrameBlock : per-frame spatial Mamba  (B*T, C, H, W), x_size=(H, W)
- GlobalBlock: T frames jointly, hybrid format
    norm/FFN  → (B, T*C, H, W)  (2D spatial 구조 유지)
    Mamba     → (B, H*W*T, C),  x_size=(H, W*T)
                 (h,w) 위치별 T프레임 연속 배치 → position-first temporal 집계
- 3-scale pyramid → PCD alignment → conv fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from vd.archs.arch_util import DCNv2Pack, Downsample, LayerNorm2d
from vd.archs.mdnet_arch import FeedForward
from vd.archs.mamba import ExtendedMamba, ScanTransform


class PCDAlignment(nn.Module):
    """Pyramid Cascading Deformable alignment."""
    def __init__(self, num_feat=64, deformable_groups=8):
        super().__init__()
        self.offset_conv1 = nn.ModuleDict()
        self.offset_conv2 = nn.ModuleDict()
        self.offset_conv3 = nn.ModuleDict()
        self.dcn_pack = nn.ModuleDict()
        self.feat_conv = nn.ModuleDict()

        for i in range(3, 0, -1):
            level = f'l{i}'
            self.offset_conv1[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
            if i == 3:
                self.offset_conv2[level] = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            else:
                self.offset_conv2[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
                self.offset_conv3[level] = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.dcn_pack[level] = DCNv2Pack(num_feat, num_feat, 3, padding=1, deformable_groups=deformable_groups)
            if i < 3:
                self.feat_conv[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)

        self.cas_offset_conv1 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
        self.cas_offset_conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.cas_dcnpack = DCNv2Pack(num_feat, num_feat, 3, padding=1, deformable_groups=deformable_groups)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu', a=0.1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, nbr_feat, ref_feat):
        upsampled_offset, upsampled_feat = None, None
        for i in range(3, 0, -1):
            level = f'l{i}'
            offset = torch.cat([nbr_feat[i - 1], ref_feat[i - 1]], dim=1)
            offset = self.leaky_relu(self.offset_conv1[level](offset))
            if i == 3:
                offset = self.leaky_relu(self.offset_conv2[level](offset))
            else:
                offset = self.leaky_relu(self.offset_conv2[level](torch.cat([offset, upsampled_offset], dim=1)))
                offset = self.leaky_relu(self.offset_conv3[level](offset))
            feat = self.dcn_pack[level](nbr_feat[i - 1], offset)
            if i < 3:
                feat = self.feat_conv[level](torch.cat([feat, upsampled_feat], dim=1))
            if i > 1:
                feat = self.leaky_relu(feat)
                upsampled_offset = F.interpolate(offset, scale_factor=2, mode='bilinear', align_corners=False) * 2
                upsampled_feat = F.interpolate(feat, scale_factor=2, mode='bilinear', align_corners=False)

        offset = torch.cat([feat, ref_feat[0]], dim=1)
        offset = self.leaky_relu(self.cas_offset_conv2(self.leaky_relu(self.cas_offset_conv1(offset))))
        return self.leaky_relu(self.cas_dcnpack(feat, offset))


# =============================================================================
# Mamba blocks (frame / global)
# =============================================================================
class FrameBlock(nn.Module):
    """Per-frame spatial Mamba. Each frame processed independently: (B*T, C, H, W)."""
    def __init__(self, dim, ffn_expansion_factor, scan_transform):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.mamba = ExtendedMamba(dim, scan_transform)
        self.norm2 = LayerNorm2d(dim)
        self.ffn   = FeedForward(dim, ffn_expansion_factor)

    def forward(self, x):
        """x: (B*T, C, H, W)"""
        _, _, H, W = x.shape
        x_pre = self.mamba(rearrange(self.norm1(x), 'n c h w -> n (h w) c'), x_size=(H, W))
        x = x + rearrange(x_pre, 'n (h w) c -> n c h w', h=H, w=W)
        return x + self.ffn(self.norm2(x))


class GlobalBlock(nn.Module):
    """
    T frames jointly as a single H*W*T sequence.
    norm/FFN: (B, T*C, H, W)  — 2D spatial 구조 유지
    Mamba:    (B, H*W*T, C),  x_size=(H, W*T)
              같은 (h,w) 위치의 T프레임이 연속 → position-first temporal 집계
    """
    def __init__(self, dim, num_frames, ffn_expansion_factor, scan_transform):
        super().__init__()
        self.T   = num_frames
        tc       = dim * num_frames
        self.norm1 = LayerNorm2d(tc)
        self.mamba = ExtendedMamba(dim, scan_transform)
        self.norm2 = LayerNorm2d(tc)
        self.ffn   = FeedForward(tc, ffn_expansion_factor)

    def forward(self, x):
        """x: (B*T, C, H, W)"""
        BT, _, H, W = x.shape
        B = BT // self.T

        x_4d  = rearrange(x, '(b t) c h w -> b (t c) h w', b=B, t=self.T)
        x_pre = rearrange(self.norm1(x_4d), 'b (t c) h w -> b (h w t) c', t=self.T)
        x_pre = self.mamba(x_pre, x_size=(H, W * self.T))
        x_4d  = x_4d + rearrange(x_pre, 'b (h w t) c -> b (t c) h w', h=H, w=W, t=self.T)
        x_4d  = x_4d + self.ffn(self.norm2(x_4d))
        return rearrange(x_4d, 'b (t c) h w -> (b t) c h w', t=self.T)

# =============================================================================
# Temporal Mamba Aggregation
# =============================================================================
class TemporalAggregationBlock(nn.Module):
    """FrameBlock → GlobalBlock 교대 num_layers회. 입출력: (B*T, C, H, W)."""
    def __init__(
        self,
        num_feat,
        input_frames=3,
        ffn_expansion_factor=2,
        num_layers=2,
        **kwargs,
    ):
        super().__init__()

        self.scan_transform = ScanTransform(scan_type='diagonal', scan_count=8, merge_method='concate')

        self.frame_blocks = nn.ModuleList([
            FrameBlock(num_feat, ffn_expansion_factor, self.scan_transform)
            for _ in range(num_layers)
        ])
        self.global_blocks = nn.ModuleList([
            GlobalBlock(num_feat, input_frames, ffn_expansion_factor, self.scan_transform)
            for _ in range(num_layers)
        ])

    def forward(self, inputs):
        x = inputs
        for f_blk, g_blk in zip(self.frame_blocks, self.global_blocks):
            x = f_blk(x)
            x = g_blk(x)
        return x


# =============================================================================
# Temporal Alignment Network (TANet)
# =============================================================================
class TemporalAlignmentNetwork(nn.Module):
    """3-scale Mamba temporal aggregation → PCD alignment → conv fusion."""
    def __init__(self, num_feat=64, input_frames=3, num_layers_per_scale=1):
        super().__init__()
        self.frames = input_frames
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # Multi-scale feature extraction
        self.branch_in1 = nn.Conv2d(num_feat, num_feat, 1)
        self.branch_in2 = nn.Conv2d(num_feat * 2, num_feat * 2, 1)
        self.branch_in3 = nn.Conv2d(num_feat * 4, num_feat * 4, 1)
        self.down1  = Downsample(num_feat, num_feat * 2, activation=True)
        self.down2  = Downsample(num_feat * 2, num_feat * 4, activation=True)

        # Multi-scale temporal aggregation
        self.tab_l1 = TemporalAggregationBlock(num_feat, input_frames, num_layers=num_layers_per_scale)
        self.tab_l2 = TemporalAggregationBlock(num_feat * 2, input_frames, num_layers=num_layers_per_scale)
        self.tab_l3 = TemporalAggregationBlock(num_feat * 4, input_frames, num_layers=num_layers_per_scale)

        self.branch_out1 = nn.Conv2d(num_feat, num_feat, 1)
        self.branch_out2 = nn.Conv2d(num_feat * 2, num_feat, 1)
        self.branch_out3 = nn.Conv2d(num_feat * 4, num_feat, 1)

        # PCD alignment (geometry correction)
        self.pcd_align = PCDAlignment(num_feat, deformable_groups=8)

        # Temporal aggregation after PCD
        self.tab_final = TemporalAggregationBlock(num_feat, input_frames, num_layers=num_layers_per_scale)

        # frame fusion conv
        self.fusion = nn.Conv2d(num_feat * input_frames, num_feat, 3, 1, 1)

    def forward(self, x):
        n = x.size(0)
        b = n // self.frames

        # Scale 1 (1x)
        feat_1 = self.leaky_relu(self.branch_in1(x))
        # feat_1 = self.tab_l1(feat_1)

        # Scale 2 (0.5x)
        feat_2 = self.leaky_relu(self.branch_in2(self.down1(feat_1)))
        # feat_2 = self.tab_l2(feat_2)

        # Scale 3 (0.25x)
        feat_3 = self.leaky_relu(self.branch_in3(self.down2(feat_2)))
        # feat_3 = self.tab_l3(feat_3)
        
        # Before PCD
        feat_1 = self.branch_out1(feat_1)
        feat_2 = self.branch_out2(feat_2)
        feat_3 = self.branch_out3(feat_3)
        
        feat_1 = rearrange(feat_1, '(b t) c h w -> b t c h w', b=b, t=self.frames)
        feat_2 = rearrange(feat_2, '(b t) c h w -> b t c h w', b=b, t=self.frames)
        feat_3 = rearrange(feat_3, '(b t) c h w -> b t c h w', b=b, t=self.frames)
        ref_feat = [feat_1[:, 1].contiguous(),
                    feat_2[:, 1].contiguous(),
                    feat_3[:, 1].contiguous()]

        # PCD alignment
        aligned_feats = []
        for i in range(self.frames):
            nbr_feat = [feat_1[:, i].contiguous(),
                        feat_2[:, i].contiguous(),
                        feat_3[:, i].contiguous()]
            aligned_feats.append(self.pcd_align(nbr_feat, ref_feat))

        # Fusion
        aligned = torch.stack(aligned_feats, dim=1)
        aligned = rearrange(aligned, 'b t c h w -> (b t) c h w')
        # aligned = self.tab_final(aligned)
        aligned = rearrange(aligned, '(b t) c h w -> b (t c) h w', b=b, t=self.frames)

        return self.leaky_relu(self.fusion(aligned))