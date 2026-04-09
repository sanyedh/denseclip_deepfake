import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from mmcv.cnn import ConvModule
from mmseg.ops import Upsample, resize
from mmseg.models.builder import LOSSES
from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.fpn_head import FPNHead
from mmseg.models.decode_heads.decode_head import BaseDecodeHead


@LOSSES.register_module()
class CustomFocalLoss(nn.Module):
    def __init__(self, use_sigmoid=False, gamma=2.0, alpha=0.5, loss_weight=1.0, ignore_index=255, **kwargs):
        super(CustomFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self, cls_score, label, **kwargs):
        # 1. 计算基础的 CrossEntropy (会自动处理 Softmax)
        logpt = -F.cross_entropy(cls_score, label, ignore_index=self.ignore_index, reduction='none')
        pt = torch.exp(logpt)

        # 2. Focal Loss 核心: (1 - pt)^gamma，直接压制简单样本(FaceSwap)
        focal_loss = ((1 - pt) ** self.gamma) * logpt

        # 3. 动态 Alpha 权重分配 (Fake 为 1, Real 为 0)
        if self.alpha is not None:
            alpha_weight = torch.where(label == 1,
                                       torch.tensor(self.alpha, device=label.device),
                                       torch.tensor(1.0 - self.alpha, device=label.device))
            focal_loss = alpha_weight * focal_loss

        # 4. 过滤掉 Padding 等无效区域
        valid_mask = (label != self.ignore_index).float()
        focal_loss = focal_loss * valid_mask

        # 5. 求均值并乘以你设定的 Loss 权重
        loss = -focal_loss.sum() / valid_mask.sum().clamp(min=1.0)
        return loss * self.loss_weight

    @property
    def loss_name(self):
        return 'loss_custom_focal'
@HEADS.register_module()
class IdentityHead(BaseDecodeHead):
    """Panoptic Feature Pyramid Networks.
    This head is the implementation of `Semantic FPN
    <https://arxiv.org/abs/1901.02446>`_.
    Args:
        feature_strides (tuple[int]): The strides for input feature maps.
            stack_lateral. All strides suppose to be power of 2. The first
            one is of largest resolution.
    """

    def __init__(self, **kwargs):
        super(IdentityHead, self).__init__(
            input_transform=None, **kwargs)
        self.conv_seg = None

    def forward(self, inputs):
        return inputs


# 在 denseclip/heads.py 中修改 CrossAttnTextGuidedFPNHead
@HEADS.register_module()
class CrossAttnTextGuidedFPNHead(FPNHead):
    # 【修改点】：增加 use_cross_attn 参数
    def __init__(self, text_channels=1024, num_heads=8, use_cross_attn=True, **kwargs):
        super(CrossAttnTextGuidedFPNHead, self).__init__(**kwargs)
        self.use_cross_attn = use_cross_attn

        # 将全局文本特征降维到 256 维，用于最终的分类点乘 (这部分必须保留)
        self.global_text_proj = nn.Linear(text_channels, self.channels)

        # ================= 消融实验逻辑 =================
        if self.use_cross_attn:
            self.text_proj = nn.Linear(text_channels, self.channels)
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=self.channels,  # 256
                num_heads=num_heads,
                batch_first=True,
                dropout=0.3
            )
            self.attn_norm = nn.LayerNorm(self.channels)
            self.gamma = nn.Parameter(torch.ones(1) * 0.01)
        # ===============================================

        self.conv_seg = nn.Identity()

    def forward(self, inputs):
        text_embeddings = inputs[0]
        text_seqs = inputs[1]
        visual_inputs = inputs[2:-1]
        key_padding_mask = inputs[-1]

        # 提取 FPN 视觉特征 [B, 256, H, W]
        feat = super(CrossAttnTextGuidedFPNHead, self).forward(visual_inputs)
        B, _, H, W = feat.shape

        # ================= 消融实验逻辑 =================
        if self.use_cross_attn:
            B_seq, K, S, C = text_seqs.shape
            key_value = text_seqs.reshape(B_seq, K * S, C)
            key_value_256 = self.text_proj(key_value)

            query = feat.flatten(2).transpose(1, 2)
            query_fp32 = query.float()
            key_value_fp32 = key_value_256.float()

            attn_output_fp32, _ = self.cross_attn(
                query=query_fp32,
                key=key_value_fp32,
                value=key_value_fp32,
                key_padding_mask=key_padding_mask
            )

            attn_output = attn_output_fp32.type_as(query)
            attn_feat_norm = self.attn_norm(attn_output)
            attn_feat = attn_feat_norm.transpose(1, 2).reshape(B, self.channels, H, W)

            enhanced_feat = feat + self.gamma * attn_feat

            if self.training and torch.rand(1).item() < 0.01:
                print(f"\n[DEBUG] Cross-Attention Gamma 活跃度: {self.gamma.item():.6f}")
        else:
            # 消融状态：不进行注意力融合，特征保持不变
            enhanced_feat = feat
        # ===============================================

        # 将全局文本特征映射到 256 维
        text_embeddings_256 = self.global_text_proj(text_embeddings)

        # 归一化后计算 Logits
        enhanced_feat = F.normalize(enhanced_feat.float(), dim=1, p=2, eps=1e-5).type_as(feat)
        text_embeddings_256 = F.normalize(text_embeddings_256.float(), dim=2, p=2, eps=1e-5).type_as(feat)

        logits = torch.einsum('bchw,bkc->bkhw', enhanced_feat, text_embeddings_256)
        logits = logits / 0.07

        return logits