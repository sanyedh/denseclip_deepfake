# ================= 优化器配置 =================
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={
            'forensics_adapter': dict(lr_mult=1.0),
            'adapter_out_proj': dict(lr_mult=2.0),

            # 【重要恢复】：为 injection_gamma 开启专属通道，使其免除正则化惩罚并快速响应
            'injection_gamma': dict(lr_mult=10.0, decay_mult=0.0),

            'decode_head.visual_proj': dict(lr_mult=10.0),
            'backbone': dict(lr_mult=0.1),
            'text_encoder': dict(lr_mult=0.0)
        }
    )
)

optimizer_config = dict()

# ================= 学习率策略 =================
lr_config = dict(
    policy='poly',
    power=0.9,
    min_lr=1e-6,
    by_epoch=False,
    warmup='linear',
    warmup_iters=3000,
    warmup_ratio=1e-6
)

# ================= 运行与评估设置 =================
runner = dict(type='IterBasedRunner', max_iters=80000)
checkpoint_config = dict(by_epoch=False, interval=8000)
evaluation = dict(interval=4000, metric='mIoU', save_best='mIoU')

# ================= 硬件优化 =================
fp16 = dict(loss_scale='dynamic')