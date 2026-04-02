import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class ForensicsAdapter(nn.Module):
    def __init__(self, vit_name='vit_tiny_patch16_224', clip_dim=1024):
        super().__init__()
        # 引入轻量级 ViT 作为旁路适配器
        self.vit_model = timm.create_model(vit_name, pretrained=True, num_classes=0)

        if self.vit_model.cls_token is not None:
            self.vit_model.pos_embed = nn.Parameter(self.vit_model.pos_embed[:, 1:, ...])
        self.vit_model.cls_token = None
        self.vit_model.norm = nn.Identity()

        self.num_features = self.vit_model.num_features # 192

        self.fusion_proj = nn.Conv2d(clip_dim, self.num_features, kernel_size=1)

        self.mask_decoder = nn.Sequential(
            nn.Conv2d(self.num_features, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 1, 1)
        )

    def intra_contra_loss(self, adapter_feat, gt_semantic_seg):
        """【终极极速版 + FP16绝对安全版】全张量化类内对比损失"""
        if gt_semantic_seg.dim() == 3:
            gt_semantic_seg = gt_semantic_seg.unsqueeze(1)

        B, D, H_p, W_p = adapter_feat.shape
        L = H_p * W_p

        # 下采样并展平
        patch_labels = F.interpolate(gt_semantic_seg.float(), size=(H_p, W_p), mode='nearest').reshape(B, L)
        embeddings = adapter_feat.reshape(B, D, L).permute(0, 2, 1)

        # ================= 【防 NaN 核心修复 1】 =================
        # 必须在 FP32 精度下归一化，且强制设定安全的 eps=1e-5
        embeddings = F.normalize(embeddings.float(), dim=-1, eps=1e-5).type_as(embeddings)
        # ========================================================

        # 掩码构建
        valid_mask = (patch_labels != 255).float()
        fake_mask = (patch_labels == 1).float() * valid_mask # [B, L]
        real_mask = (patch_labels == 0).float() * valid_mask # [B, L]

        # ================= 【防 NaN 核心修复 2】 =================
        # 将极易引起溢出/下溢的对比运算提升至局部 FP32 空间执行
        embeddings_fp32 = embeddings.float()
        fake_mask_fp32 = fake_mask.float()
        real_mask_fp32 = real_mask.float()

        # [B, L, L] 批量计算全量 Token 相似度
        sim_matrix = torch.bmm(embeddings_fp32, embeddings_fp32.transpose(1, 2)) / 0.1
        exp_sim = torch.exp(sim_matrix)

        # 【核心提速】：利用张量广播和批量矩阵乘法生成对比掩码，替代 for 循环
        pos_fake_mask = torch.bmm(fake_mask_fp32.unsqueeze(2), fake_mask_fp32.unsqueeze(1))
        pos_real_mask = torch.bmm(real_mask_fp32.unsqueeze(2), real_mask_fp32.unsqueeze(1))
        neg_mask = torch.bmm(fake_mask_fp32.unsqueeze(2), real_mask_fp32.unsqueeze(1)) + \
                   torch.bmm(real_mask_fp32.unsqueeze(2), fake_mask_fp32.unsqueeze(1))

        # 直接沿特征维度并行求和
        l_pos_fake = (exp_sim * pos_fake_mask).sum(dim=2) # [B, L]
        l_pos_real = (exp_sim * pos_real_mask).sum(dim=2) # [B, L]
        l_neg = (exp_sim * neg_mask).sum(dim=2) # [B, L]

        # ================= 【防 NaN 核心修复 3】 =================
        # 并行计算对数损失 (所有 1e-8 提升至 FP16 安全值 1e-5)
        loss_fake = -torch.log(l_pos_fake / (l_neg + l_pos_fake + 1e-5) + 1e-5)
        loss_real = -torch.log(l_pos_real / (l_neg + l_pos_real + 1e-5) + 1e-5)

        # 屏蔽无效像素的损失
        loss_fake = loss_fake * fake_mask_fp32
        loss_real = loss_real * real_mask_fp32

        # 计算单张图片包含的真假像素点数量，安全下限设为 1e-5
        num_fake = fake_mask_fp32.sum(dim=1).clamp(min=1e-5) # [B]
        num_real = real_mask_fp32.sum(dim=1).clamp(min=1e-5) # [B]

        # 算清单张图片的平均 Loss
        batch_loss_fake = loss_fake.sum(dim=1) / num_fake
        batch_loss_real = loss_real.sum(dim=1) / num_real

        # 使用布尔掩码判断单张图片是否具有可对比的正负样本
        valid_batch = ((fake_mask_fp32.sum(dim=1) > 0) & (real_mask_fp32.sum(dim=1) > 0)).float()

        # 对符合要求的图片 Loss 汇总计算总平均值
        total_loss = ((batch_loss_fake + batch_loss_real) * valid_batch).sum() / valid_batch.sum().clamp(min=1e-5)

        # 算完转回原始张量类型（FP16），确保显存和后续计算兼容
        return total_loss.type_as(adapter_feat)

    def forward(self, img, clip_feature_stage3, gt_semantic_seg=None):
        B, C, H, W = img.shape

        if hasattr(self.vit_model.patch_embed, 'img_size'):
            self.vit_model.patch_embed.img_size = (H, W)

        x = self.vit_model.patch_embed(img)
        H_p, W_p = H // 16, W // 16

        pos_embed = self.vit_model.pos_embed
        grid_size = int(pos_embed.shape[1] ** 0.5)
        pos_embed = pos_embed.permute(0, 2, 1).reshape(1, self.num_features, grid_size, grid_size)
        pos_embed = F.interpolate(pos_embed, size=(H_p, W_p), mode='bilinear', align_corners=False)
        pos_embed = pos_embed.reshape(1, self.num_features, -1).permute(0, 2, 1)
        x = x + pos_embed

        clip_f = self.fusion_proj(clip_feature_stage3)
        clip_f_seq = clip_f.flatten(2).permute(0, 2, 1)

        for i, block in enumerate(self.vit_model.blocks):
            x = block(x)
            if i < 4:
                x = x + clip_f_seq

        spatial_feat = x.permute(0, 2, 1).reshape(B, self.num_features, H_p, W_p)
        xray_pred = self.mask_decoder(spatial_feat)

        loss_intra = torch.tensor(0.0).to(img.device)
        if gt_semantic_seg is not None:
            loss_intra = self.intra_contra_loss(spatial_feat, gt_semantic_seg)

        return spatial_feat, xray_pred, loss_intra