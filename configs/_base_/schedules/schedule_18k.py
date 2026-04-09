# ================= 优化器配置 =================
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={
            'contexts': dict(lr_mult=6.0),
            'gamma': dict(lr_mult=6.0, decay_mult=0.0),
            'forensics_adapter': dict(lr_mult=0.5),
            'adapter_out_proj': dict(lr_mult=1.0),
            'injection_gamma': dict(lr_mult=6.0, decay_mult=0.0),
            'decode_head.text_proj': dict(lr_mult=2.0),
            'decode_head.global_text_proj': dict(lr_mult=2.0),
            'decode_head.gamma': dict(lr_mult=1.0, decay_mult=0.0),
            'backbone': dict(lr_mult=0.1),
            'text_encoder': dict(lr_mult=0.0)
        }
    )
)

# 1. 优化器
optimizer_config = dict(
    type='GradientCumulativeOptimizerHook',
    cumulative_iters=1,
    grad_clip=dict(max_norm=35, norm_type=2)
)

# 2. 学习率策略
lr_config = dict(
    policy='poly',
    power=0.9,
    min_lr=1e-6,
    by_epoch=False,
    warmup='linear',
    # 【修改】：总步数调回 18k，预热期也相应缩回 1000 步（这能让模型更快进入正式学习状态，迎接 10k 左右的巅峰）
    warmup_iters=1000,
    warmup_ratio=1e-6
)

# 3. 运行与评估设置
# 【修改】：总步数锁定为 18000 步
runner = dict(type='IterBasedRunner', max_iters=18000)

# 【修改】：评估和保存间隔改为 1500 步。
# 这样在达到 10000 步之前，你会有 6 次验证机会（1500, 3000, 4500, 6000, 7500, 9000），能非常精准地捕捉到性能最高的瞬间。
checkpoint_config = dict(by_epoch=False, interval=1500)
evaluation = dict(interval=1500, metric='mIoU', save_best='mIoU')