import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from mmcv.cnn import ConvModule
from mmseg.ops import Upsample, resize

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.fpn_head import FPNHead
from mmseg.models.decode_heads.decode_head import BaseDecodeHead


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


@HEADS.register_module()
class CrossAttnTextGuidedFPNHead(FPNHead):
    """终极优化版：在 256 维空间直接进行跨模态对齐，并引入 Padding Mask 与 LayerNorm 防止注意力污染和数值爆炸"""
    def __init__(self, text_channels=1024, num_heads=8, **kwargs):
        super(CrossAttnTextGuidedFPNHead, self).__init__(**kwargs)

        # 1. 文本序列特征降维：用于高效计算 Cross-Attention (Query-Key-Value)
        self.text_proj = nn.Linear(text_channels, self.channels)

        # 2. 跨模态注意力机制 (在 256 维空间计算)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.channels,  # 256
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )

        # 3. 【新增】：LayerNorm 层，用于驯服注意力输出的极值，防止冲垮视觉主干特征
        self.attn_norm = nn.LayerNorm(self.channels)

        self.gamma = nn.Parameter(torch.ones(1) * 0.01)

        # 4. 将全局文本特征降维到 256 维，用于最终的分类点乘
        self.global_text_proj = nn.Linear(text_channels, self.channels)

        # 覆盖父类自带的分类层
        self.conv_seg = nn.Identity()

    def forward(self, inputs):
        # 接收从 denseclip.py 传来的打包变量
        text_embeddings = inputs[0]   # 全局特征 [B, K=2, 1024]
        text_seqs = inputs[1]         # 序列特征 [B, K=2, S长度, 1024]
        visual_inputs = inputs[2:-1]  # 提取 FPN 四层视觉特征
        key_padding_mask = inputs[-1] # 提取 Padding Mask [B, K*S]

        B, K, S, C = text_seqs.shape

        # 文本序列特征降维 [B, K*S, 256]
        key_value = text_seqs.reshape(B, K * S, C)
        key_value_256 = self.text_proj(key_value)

        # 提取 FPN 视觉特征 [B, 256, H, W]
        feat = super(CrossAttnTextGuidedFPNHead, self).forward(visual_inputs)

        _, _, H, W = feat.shape
        query = feat.flatten(2).transpose(1, 2) # [B, H*W, 256]

        # ==================== 【防 NaN 核心区 1：FP32 注意力计算】 ====================
        # 强制转为 float32，防止 Attention 内部 Softmax 在 FP16 下溢出
        query_fp32 = query.float()
        key_value_fp32 = key_value_256.float()

        attn_output_fp32, _ = self.cross_attn(
            query=query_fp32,
            key=key_value_fp32,
            value=key_value_fp32,
            key_padding_mask=key_padding_mask
        )

        # 安全计算完后再转回原始类型
        attn_output = attn_output_fp32.type_as(query)
        # =========================================================================

        # LayerNorm
        attn_feat_norm = self.attn_norm(attn_output)

        # reshape 回空间维度 [B, 256, H, W]
        attn_feat = attn_feat_norm.transpose(1, 2).reshape(B, self.channels, H, W)

        # 残差注入
        enhanced_feat = feat + self.gamma * attn_feat

        # 将全局文本特征映射到 256 维
        text_embeddings_256 = self.global_text_proj(text_embeddings)

        # ==================== 【防 NaN 核心区 2：FP32 安全归一化】 ====================
        # 必须强制设置 eps=1e-5，并且在 float32 精度下归一化，彻底杜绝除零错误！
        enhanced_feat = F.normalize(enhanced_feat.float(), dim=1, p=2, eps=1e-5).type_as(feat)
        text_embeddings_256 = F.normalize(text_embeddings_256.float(), dim=2, p=2, eps=1e-5).type_as(feat)
        # =========================================================================

        # 点乘计算 Logits：bchw 与 bkc -> bkhw
        logits = torch.einsum('bchw,bkc->bkhw', enhanced_feat, text_embeddings_256)

        # 放大对比度 (CLIP 默认的 temperature scaling)
        logits = logits / 0.07

        if self.training and torch.rand(1).item() < 0.01:
            print(f"\n[DEBUG] Cross-Attention Gamma 活跃度: {self.gamma.item():.6f}")

        return logits