import torch
from torch import Tensor
from torch import nn as nn
from torch.nn import functional as F
from torchvision import models, transforms
from torchvision.models.feature_extraction import create_feature_extractor
from vd.utils.registry import LOSS_REGISTRY
from .loss_util import weighted_loss, _reduction_modes
from .loss_arch import VGGFeatureExtractor

@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')

@LOSS_REGISTRY.register()
class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        return self.loss_weight * l1_loss(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class PerceptualLoss(nn.Module):
    """Perceptual loss with commonly used style loss.

    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculating losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 0.
        criterion (str): Criterion used for perceptual loss. Default: 'l1'.
    """

    def __init__(self,
                 layer_weights,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 range_norm=False,
                 perceptual_weight=1.0,
                 style_weight=0.,
                 criterion='l1'):
        super().__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm)

        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.MSELoss()
        elif self.criterion_type == 'fro':
            self.criterion = None
        else:
            raise NotImplementedError(f'{criterion} criterion has not been supported.')

    def forward(self, x, gt):
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    percep_loss += torch.norm(x_features[k] - gt_features[k], p='fro') * self.layer_weights[k]
                else:
                    percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        # calculate style loss
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    style_loss += torch.norm(
                        self._gram_mat(x_features[k]) - self._gram_mat(gt_features[k]), p='fro') * self.layer_weights[k]
                else:
                    style_loss += self.criterion(self._gram_mat(x_features[k]), self._gram_mat(
                        gt_features[k])) * self.layer_weights[k]
            style_loss *= self.style_weight
        else:
            style_loss = None

        return percep_loss, style_loss

    def _gram_mat(self, x):
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram

@LOSS_REGISTRY.register()
class DecompositionContrastiveLoss(nn.Module):
    """
    VGG feature-space InfoNCE decomposition contrastive loss.

    VGG16 relu3_3 (features[:16]) 출력 feature map에 channel-wise L2-norm 후
    위치별 cosine similarity를 공간 평균하여 거리 계산.
    (contrastive-demoire 참조: vgg16 block 2 = features[9:16])

    anchor   = vgg(minmax(moire[t]))            — 순수 moire 예측
    positive = vgg(minmax(lq[t] - gt[t]))       — AdaIN(lq,gt) - gt = moire 잔차
    negative = vgg(gt[t']), t'=0..T-1
              + vgg(gt[b', t]), b'≠b  (in_batch_negative=True 시)

    loss_t = CrossEntropy([sim(anchor, pos)/τ, sim(anchor, neg[0])/τ, ...])

    Notes:
    - gt/lq 즉시 detach; gradient는 moire → vgg activation → anchor 경로로만 흐름.
    - lq에는 호출 측에서 AdaIN(lq_orig, gt)를 전달. lq - gt = moire 잔차.
    - VGG16 파라미터 frozen, ImageNet 정규화 적용, 항상 eval 모드 유지.

    Args:
        loss_weight (float): 전체 loss 가중치. Default: 1.0.
        temperature (float): InfoNCE temperature. Default: 0.07.
        in_batch_negative (bool): 배치 내 타 샘플 gt를 negative에 추가. Default: True.
    """
    def __init__(
        self,
        loss_weight: float = 1.0,
        temperature: float = 0.07,
        in_batch_negative: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self.temperature = temperature
        self.in_batch_negative = in_batch_negative

        vgg = models.vgg16(pretrained=True)
        self.vgg = nn.Sequential(*list(vgg.features[:16]))
        for p in self.vgg.parameters():
            p.requires_grad_(False)
        self.vgg.eval()

        self.register_buffer('vgg_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('vgg_std',  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def train(self, mode: bool = True):
        super().train(mode)
        self.vgg.eval()  # 학습 모드 전환 시에도 VGG는 항상 eval 유지
        return self

    def _minmax(self, x: torch.Tensor) -> torch.Tensor:
        """per-sample min-max → [0, 1]. gradient 유지."""
        x_flat = x.flatten(1)
        lo = x_flat.min(dim=1)[0].view(-1, 1, 1, 1)
        hi = x_flat.max(dim=1)[0].view(-1, 1, 1, 1)
        return (x - lo) / (hi - lo + 1e-8)

    def _extract(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, minmax: bool = False) -> torch.Tensor:
        """(B, 3, H, W) → VGG relu3_3 → channel L2-norm → (B, 256, H', W')"""
        if minmax:
            x = self._minmax(x)
        feat = self.vgg((x - mean) / std)           # (B, 256, H/4, W/4)
        return F.normalize(feat, p=2, dim=1)        # channel-wise L2-norm

    def _infonce(
        self,
        q:    torch.Tensor,
        p:    torch.Tensor,
        negs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            q, p (Tensor): (B, C, H, W) — channel L2-normalized feature maps.
            negs (Tensor): (B, K, C, H, W) — channel L2-normalized feature maps.
        Returns:
            (Tensor): scalar InfoNCE loss.
        """
        sim_pos = (q * p).sum(dim=1).mean(dim=(-2, -1)).unsqueeze(1) / self.temperature       # (B, 1)
        sim_neg = (negs * q.unsqueeze(1)).sum(dim=2).mean(dim=(-2, -1)) / self.temperature    # (B, K)
        logits  = torch.cat([sim_pos, sim_neg], dim=1)                                         # (B, 1+K)
        labels  = torch.zeros(q.size(0), dtype=torch.long, device=q.device)
        return F.cross_entropy(logits, labels)

    def forward(
        self,
        moire: torch.Tensor,
        lq:    torch.Tensor,
        gt:    torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            moire (Tensor): (B, T, 3, H, W) — gradient 흐름.
            lq    (Tensor): (B, T, 3, H, W) — positive.
            gt    (Tensor): (B, T, 3, H, W) — negative.
        Returns:
            (Tensor): scalar decomposition loss.
        """
        gt = gt.detach()
        lq = lq.detach()
        B, T, C, H, W = moire.shape

        mean = self.vgg_mean.to(moire)
        std  = self.vgg_std.to(moire)

        with torch.no_grad():
            # positive: AdaIN(lq, gt) - gt = moire 잔차 (minmax로 범위 정규화)
            res_flat = self._extract((lq - gt).view(B * T, C, H, W), mean, std, minmax=True)
            gt_flat  = self._extract(gt.view(B * T, C, H, W), mean, std, minmax=True)
            _, Cf, Hf, Wf = gt_flat.shape
            pos_feats = res_flat.view(B, T, Cf, Hf, Wf)
            gt_feats  = gt_flat.view(B, T, Cf, Hf, Wf)

        terms = []
        for t in range(T):
            q = self._extract(moire[:, t], mean, std, minmax=True)   # anchor: 순수 moire 예측
            p = pos_feats[:, t]                            # positive: moire 잔차

            negs = gt_feats                                # (B, T, C, H', W') — 클린 negative
            if self.in_batch_negative and B > 1:
                mask  = ~torch.eye(B, dtype=torch.bool, device=q.device)
                ib_gt = gt_feats[:, t].unsqueeze(0).expand(B, -1, -1, -1, -1)[mask].view(B, B - 1, *q.shape[1:])
                negs  = torch.cat([negs, ib_gt], dim=1)   # (B, T+B-1, C, H', W')

            terms.append(self._infonce(q, p, negs))

        return self.loss_weight * torch.stack(terms).mean()