# ================= 优化器配置 =================
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={
            'contexts': dict(lr_mult=10.0),

            # 【新增】：denseclip 主类里的文本特征残差缩放参数 gamma，也给 10 倍加速起步
            'gamma': dict(lr_mult=10.0, decay_mult=0.0),

            'forensics_adapter': dict(lr_mult=1.0),
            'adapter_out_proj': dict(lr_mult=2.0),
            'injection_gamma': dict(lr_mult=10.0, decay_mult=0.0),

            # 【修改这里】：原来的 visual_to_text_proj 已经删了，
            # 换成我们新建的两个降维线性层，同样给它们 10 倍学习率加速收敛
            'decode_head.text_proj': dict(lr_mult=2.0),
            'decode_head.global_text_proj': dict(lr_mult=2.0),

            'decode_head.gamma': dict(lr_mult=1.0, decay_mult=0.0),

            'backbone': dict(lr_mult=0.1),
            'text_encoder': dict(lr_mult=0.0)
        }
    )
)

# 【核心修改区】：引入梯度累加来拯救显存
# 假设你在 denseclip_deepfake_r50.py 中把 samples_per_gpu 改成了 4
# 那么 cumulative_iters 设为 2，等效 Batch Size = 4 * 2 = 8 (保持原有的训练节奏)
# 如果你把 samples_per_gpu 改成了 2 才能跑起来，那么这里就改成 4 (2 * 4 = 8)
optimizer_config = dict(
    type='GradientCumulativeOptimizerHook',
    cumulative_iters=2
)

# ================= 学习率策略 =================
lr_config = dict(
    policy='poly',
    power=0.9,
    min_lr=1e-6,
    by_epoch=False,
    warmup='linear',
    warmup_iters=1500,     # 【修改】：总轮数变少，Warmup相应缩短到 1500 次
    warmup_ratio=1e-6
)

# ================= 运行与评估设置 =================
runner = dict(type='IterBasedRunner', max_iters=32000) # 【修改】：总迭代次数改为 32000
checkpoint_config = dict(by_epoch=False, interval=4000) # 【修改】：每 4000 次保存一个常规权重（防断电）
evaluation = dict(interval=1000, metric='mIoU', save_best='mIoU') # 【修改】：每 1000 次进行一次评估，更精细地捕捉最高点

# ================= 硬件优化 =================
# fp16 = dict(loss_scale='dynamic')