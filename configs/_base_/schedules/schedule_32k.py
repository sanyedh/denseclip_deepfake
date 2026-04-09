# ================= 优化器配置 =================
# (保持不变)
optimizer = dict(
    type='AdamW',
    lr=0.00005,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={
            'contexts': dict(lr_mult=6.0),
            'gamma': dict(lr_mult=6.0, decay_mult=0.0),
            'forensics_adapter': dict(lr_mult=1.0),
            'adapter_out_proj': dict(lr_mult=2.0),
            'injection_gamma': dict(lr_mult=6.0, decay_mult=0.0),
            'decode_head.text_proj': dict(lr_mult=2.0),
            'decode_head.global_text_proj': dict(lr_mult=2.0),
            'decode_head.gamma': dict(lr_mult=1.0, decay_mult=0.0),
            'backbone': dict(lr_mult=0.1),
            'text_encoder': dict(lr_mult=0.0)
        }
    )
)

optimizer_config = dict(
    type='GradientCumulativeOptimizerHook',
    cumulative_iters=2,
    grad_clip=dict(max_norm=35, norm_type=2)
)

# ================= 学习率策略 =================
lr_config = dict(
    policy='poly',
    power=0.9,
    min_lr=1e-6,
    by_epoch=False,
    warmup='linear',
    # 【修改】：总步数32000下，预热 1000 步（正好约 1 个 Epoch），占比约 3%，非常健康的比例。
    warmup_iters=1000,
    warmup_ratio=1e-6
)

# ================= 运行与评估设置 =================
# 【修改】：严格保持 32000 步！(修复了你刚才漏掉的一个 0)
runner = dict(type='IterBasedRunner', max_iters=32000)

# 【修改】：总步数不长，可以每 4000 步（约 4 个 Epoch）保存一次权重，防止磁盘撑爆。
checkpoint_config = dict(by_epoch=False, interval=4000)

# 【修改】：1个Epoch是993步。设置评估间隔为 1000，意思是模型每看完整遍数据集，就去考一次试，精度最高。
evaluation = dict(interval=1000, metric='mIoU', save_best='mIoU')