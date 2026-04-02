import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.core import add_prefix
from mmseg.ops import resize
from mmseg.models import builder
from mmseg.models.builder import SEGMENTORS
from mmseg.models.segmentors.base import BaseSegmentor
from .untils import tokenize

@SEGMENTORS.register_module()
class DenseCLIP(BaseSegmentor):
    def __init__(self, backbone, text_encoder, context_decoder, decode_head, class_names,
                 context_length, context_feature='attention', score_concat_index=3,
                 text_head=False, neck=None, tau=0.07, auxiliary_head=None, identity_head=None,
                 train_cfg=None, test_cfg=None, pretrained=None, init_cfg=None,
                 token_embed_dim=512, text_dim=1024, **args):
        super(DenseCLIP, self).__init__(init_cfg)

        if pretrained is not None:
            if backbone.get('pretrained') is None:
                backbone.pretrained = pretrained
            if text_encoder.get('pretrained') is None:
                if 'RN50' not in pretrained and 'RN101' not in pretrained and 'ViT-B' not in pretrained:
                    print('not CLIP pre-trained weight, using CLIP ViT-B-16')
                    text_encoder.pretrained = 'pretrained/ViT-B-16.pt'
                else:
                    text_encoder.pretrained = pretrained

        self.backbone = builder.build_backbone(backbone)
        self.text_encoder = builder.build_backbone(text_encoder)

        for param in self.text_encoder.parameters():
            param.requires_grad = False

        self.context_decoder = builder.build_backbone(context_decoder)
        self.context_length = context_length
        self.score_concat_index = score_concat_index
        self.context_feature = context_feature
        self.text_head = text_head
        self.tau = tau

        if neck is not None:
            self.neck = builder.build_neck(neck)

        self.num_classes = 2
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.with_identity_head = False
        self.identity_head = None
        self._init_identity_head(identity_head)

        from .forensics_adapter import ForensicsAdapter
        self.with_adapter = True
        self.forensics_adapter = ForensicsAdapter(vit_name='vit_tiny_patch16_224', clip_dim=1024)

        self.adapter_out_proj = nn.Conv2d(192, 1024, kernel_size=1)
        nn.init.constant_(self.adapter_out_proj.weight, 0)
        nn.init.constant_(self.adapter_out_proj.bias, 0)
        self.injection_gamma = nn.Parameter(torch.ones(1))

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.contexts = nn.Parameter(torch.randn(1, context_length, token_embed_dim))
        nn.init.trunc_normal_(self.contexts)
        self.gamma = nn.Parameter(torch.ones(text_dim) * 1e-4)

        assert self.with_decode_head

    def _init_decode_head(self, decode_head):
        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = 2

    def _init_auxiliary_head(self, auxiliary_head):
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(builder.build_head(head_cfg))
            else:
                self.auxiliary_head = builder.build_head(auxiliary_head)

    def _init_identity_head(self, identity_head):
        if identity_head is not None:
            self.with_identity_head = True
            self.identity_head = builder.build_head(identity_head)

    def extract_feat(self, img, gt_semantic_seg=None):
        loss_intra = torch.tensor(0.0, device=img.device)
        loss_xray = torch.tensor(0.0, device=img.device)

        if getattr(self, 'with_adapter', False):
            def injection_func(l3_feat):
                nonlocal loss_intra, loss_xray
                boundary_mask = None
                gt_seg = None

                if gt_semantic_seg is not None:
                    if gt_semantic_seg.dim() == 3:
                        gt_seg = gt_semantic_seg.unsqueeze(1)
                    else:
                        gt_seg = gt_semantic_seg

                    valid_mask = (gt_seg != 255).float()
                    clean_mask = (gt_seg == 1).float()

                    kernel_size = 15
                    padding = kernel_size // 2
                    dilation = F.max_pool2d(clean_mask, kernel_size=kernel_size, stride=1, padding=padding)
                    erosion = -F.max_pool2d(-clean_mask, kernel_size=kernel_size, stride=1, padding=padding)

                    boundary = dilation - erosion
                    boundary_mask = boundary.clone()
                    boundary_mask[valid_mask == 0] = 255
                    boundary_mask = boundary_mask.long()

                adapter_feat, xray_pred, intra = self.forensics_adapter(img, l3_feat.detach(), gt_seg)
                loss_intra = intra

                if boundary_mask is not None:
                    gt_mask_down = F.interpolate(boundary_mask.float(), size=xray_pred.shape[2:], mode='nearest')
                    valid_mask_down = (gt_mask_down != 255).float()
                    gt_mask_down = torch.clamp(gt_mask_down, 0, 1)

                    num_pos = (gt_mask_down * valid_mask_down).sum().clamp(min=1.0)
                    num_neg = ((1.0 - gt_mask_down) * valid_mask_down).sum()
                    pos_weight = torch.tensor([num_neg / num_pos], device=img.device).clamp(max=15.0)

                    bce_loss = F.binary_cross_entropy_with_logits(xray_pred, gt_mask_down, pos_weight=pos_weight, reduction='none')
                    loss_xray = (bce_loss * valid_mask_down).sum() / valid_mask_down.sum().clamp(min=1.0)

                adapter_injection = self.adapter_out_proj(adapter_feat)
                if self.training and torch.rand(1).item() < 0.01:
                    print(f"\n[DEBUG] Adapter Injection Gamma 强度: {self.injection_gamma.item():.6f}")
                return l3_feat + self.injection_gamma * adapter_injection

            x = self.backbone(img, injection_func=injection_func)
            return x, loss_intra, loss_xray
        else:
            x = self.backbone(img)
            return x, loss_intra, loss_xray

    def after_extract_feat(self, x, text_prompts=None):
        x_orig = list(x[0:4])
        global_feat, visual_embeddings = x[4]
        B, C, H, W = visual_embeddings.shape

        if self.context_feature == 'attention':
            visual_context = torch.cat([global_feat.reshape(B, C, 1), visual_embeddings.reshape(B, C, H*W)], dim=2).permute(0, 2, 1)
        else:
            visual_context = None

        flat_prompts = []
        for p in text_prompts:
            flat_prompts.extend(p)

        max_seq_len = self.text_encoder.context_length - self.context_length
        text_inputs = torch.cat([tokenize(p, context_length=max_seq_len, truncate=True) for p in flat_prompts]).to(global_feat.device)

        # ======================= 【核心修复区】 =======================
        # 1. 获取基础的文本 Padding Mask (0 是 Padding 的索引)
        base_mask = (text_inputs == 0) # 形状: [B*2, 72]

        # 2. 为可学习的 Context Tokens 创建 Mask
        # (因为它们是有效的特征，所以 Mask 值为 False，表示“不忽略”)
        context_mask = torch.zeros((len(flat_prompts), self.context_length),
                                   dtype=torch.bool, device=global_feat.device) # 形状: [B*2, 5]

        # 3. 严格对齐 CLIPTextContextEncoder 插入特征的逻辑：
        # 它是在索引 0 (SOS token) 之后，插入 context tokens
        key_padding_mask = torch.cat([
            base_mask[:, 0:1],   # 索引 0: SOS Token
            context_mask,        # 插入 5 个 Context Tokens
            base_mask[:, 1:]     # 剩余的 71 个 Tokens (含 Padding)
        ], dim=1)                # 拼接后总长度变为 77

        # 4. 展平为 [B, K*S] (如 [6, 154])，完美匹配 Heads 里的张量
        key_padding_mask = key_padding_mask.view(B, -1)
        # =============================================================

        ctx = self.contexts.expand(len(flat_prompts), -1, -1)

        text_features, text_seq_features = self.text_encoder(text_inputs, ctx)
        text_embeddings = text_features.view(B, 2, C)

        _, S, C_seq = text_seq_features.shape
        text_seq_features = text_seq_features.view(B, 2, S, C_seq)

        text_diff = self.context_decoder(text_embeddings, visual_context)
        text_embeddings = text_embeddings + self.gamma * text_diff

        visual_embeddings = F.normalize(visual_embeddings, dim=1, p=2)
        text = F.normalize(text_embeddings, dim=2, p=2)

        score_map = torch.einsum('bchw,bkc->bkhw', visual_embeddings, text)
        padded_score_map = score_map

        concat_feat = x_orig[self.score_concat_index]
        x_orig[self.score_concat_index] = torch.cat([concat_feat, padded_score_map], dim=1)

        # 统一返回 5 个变量（追加了 key_padding_mask）
        return text_embeddings, text_seq_features, x_orig, score_map, key_padding_mask

    def forward_train(self, img, img_metas, gt_semantic_seg):
        text_prompts = [meta['text_prompts'] for meta in img_metas]

        x, loss_intra, loss_xray = self.extract_feat(img, gt_semantic_seg)
        _x_orig = [x[i] for i in range(4)]

        # 【核心修改】：接收 5 个返回值
        text_embeddings, text_seq_features, x_orig, score_map, key_padding_mask = self.after_extract_feat(x, text_prompts)

        if self.with_neck:
            x_orig = list(self.neck(x_orig))
            _x_orig = x_orig

        losses = dict()

        # 【核心修改】：将 key_padding_mask 放在列表最后传给 Head
        if self.text_head:
            x = [text_embeddings, text_seq_features] + x_orig + [key_padding_mask]
        else:
            x = x_orig

        loss_decode = self._decode_head_forward_train(x, img_metas, gt_semantic_seg)
        losses.update(loss_decode)

        if self.with_identity_head:
            loss_identity = self._identity_head_forward_train(score_map/self.tau, img_metas, gt_semantic_seg)
            losses.update(loss_identity)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(_x_orig, img_metas, gt_semantic_seg)
            losses.update(loss_aux)

        if getattr(self, 'with_adapter', False):
            losses['loss_adapter_intra'] = loss_intra * 0.05
            losses['loss_adapter_xray'] = loss_xray * 0.05

        return losses

    def encode_decode(self, img, img_metas):
        text_prompts = [meta['text_prompts'] for meta in img_metas]

        x, _, _ = self.extract_feat(img, gt_semantic_seg=None)
        _x_orig = [x[i] for i in range(4)]

        # 【核心修改】：测试时同样接收 5 个返回值
        text_embeddings, text_seq_features, x_orig, score_map, key_padding_mask = self.after_extract_feat(x, text_prompts)

        if self.with_neck:
            x_orig = list(self.neck(x_orig))

        # 【核心修改】：测试时同样打包 key_padding_mask
        if self.text_head:
            x = [text_embeddings, text_seq_features] + x_orig + [key_padding_mask]
        else:
            x = x_orig

        out = self._decode_head_forward_test(x, img_metas)
        out = resize(input=out, size=img.shape[2:], mode='bilinear', align_corners=self.align_corners)
        return out

    def _decode_head_forward_train(self, x, img_metas, gt_semantic_seg):
        losses = dict()
        loss_decode = self.decode_head.forward_train(x, img_metas, gt_semantic_seg, self.train_cfg)
        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _decode_head_forward_test(self, x, img_metas):
        return self.decode_head.forward_test(x, img_metas, self.test_cfg)

    def _auxiliary_head_forward_train(self, x, img_metas, gt_semantic_seg):
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.forward_train(x, img_metas, gt_semantic_seg, self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.forward_train(x, img_metas, gt_semantic_seg, self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))
        return losses

    def _identity_head_forward_train(self, x, img_metas, gt_semantic_seg):
        losses = dict()
        loss_aux = self.identity_head.forward_train(x, img_metas, gt_semantic_seg, self.train_cfg)
        losses.update(add_prefix(loss_aux, 'aux_identity'))
        return losses

    def forward_dummy(self, img):
        dummy_meta = [{'text_prompts': ["A real authentic face.", "A forged face."]}]
        return self.encode_decode(img, dummy_meta)

    def simple_test(self, img, img_meta, rescale=True):
        seg_logit = self.inference(img, img_meta, rescale)
        seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            return seg_pred.unsqueeze(0)
        return list(seg_pred.cpu().numpy())

    def aug_test(self, imgs, img_metas, rescale=True):
        assert rescale
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            seg_logit += self.inference(imgs[i], img_metas[i], rescale)
        seg_logit /= len(imgs)
        seg_pred = seg_logit.argmax(dim=1)
        return list(seg_pred.cpu().numpy())

    def inference(self, img, img_meta, rescale):
        seg_logit = self.whole_inference(img, img_meta, rescale)
        output = F.softmax(seg_logit, dim=1)

        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3, ))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2, ))
        return output

    def whole_inference(self, img, img_meta, rescale):
        seg_logit = self.encode_decode(img, img_meta)
        if rescale:
            if not torch.onnx.is_in_onnx_export():
                img_shape = img_meta[0]['img_shape'][:2]
                seg_logit = seg_logit[:, :, :img_shape[0], :img_shape[1]]
                size = img_meta[0]['ori_shape'][:2]
            else:
                size = img.shape[2:]
            seg_logit = resize(seg_logit, size=size, mode='bilinear', align_corners=self.align_corners, warning=False)
        return seg_logit