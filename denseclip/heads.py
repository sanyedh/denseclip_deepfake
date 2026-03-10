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
class TextGuidedFPNHead(FPNHead):
    """基于动态文本 Prompt 引导的 FPN 像素级分类头"""
    def __init__(self, text_channels=1024, **kwargs):
        super(TextGuidedFPNHead, self).__init__(**kwargs)

        # 1. 新增一个映射层，将 FPN 输出的 256 维视觉特征映射到文本特征空间 (1024维)
        self.visual_proj = nn.Conv2d(self.channels, text_channels, kernel_size=1)

        # 2. 【核心修复技巧】：替换掉父类自带的纯视觉分类卷积层
        # 将其设置为 Identity (透传)，这样后面调用 super().forward() 时，
        # 就不会执行随机初始化的分类器，而是直接返回融合后的 256 维 FPN 视觉特征
        self.conv_seg = nn.Identity()

    def forward(self, inputs):
        # 开启 text_head=True 后，inputs 的第 0 个元素是动态文本特征
        text_embeddings = inputs[0] # 形状: [B, 2类, 1024]
        visual_inputs = inputs[1:]  # FPN 的四层视觉特征 [f1, f2, f3, f4]

        # 1. 调用父类 forward 提取 FPN 融合特征
        # (此时因为 conv_seg 是 Identity，返回的是纯粹的 256 维视觉特征)
        feat = super(TextGuidedFPNHead, self).forward(visual_inputs)

        # 2. 将视觉特征投影到与文本相同的维度
        feat = self.visual_proj(feat) # 形状: [B, 1024, H, W]

        # 3. L2 归一化 (统一到同一个超球面空间)
        feat = F.normalize(feat, dim=1, p=2)
        text_embeddings = F.normalize(text_embeddings, dim=2, p=2)

        # 4. 【核心】视觉像素特征与动态文本特征直接做点乘计算 Logits
        # bchw (视觉) 与 bkc (文本，k=2类) 点乘得到 bkhw (分割预测图)
        logits = torch.einsum('bchw,bkc->bkhw', feat, text_embeddings)

        # 5. 温度系数放大 (与 CLIP 保持一致，拉大对比度便于 Softmax 优化)
        logits = logits / 0.07

        return logits