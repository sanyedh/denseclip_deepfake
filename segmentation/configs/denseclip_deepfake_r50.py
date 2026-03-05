# 1. 引用基础配置
_base_ = [
    '_base_/models/denseclip_r50.py',
    '_base_/default_runtime.py',
    '_base_/schedules/schedule_80k.py'
]

# 2. 注册自定义 Dataset 模块 (确保 deepfake.py 能够被导入)
custom_imports = dict(imports=['mmseg_custom.datasets.deepfake'], allow_failed_imports=False)

# 3. 定义数据路径
data_root = r'F:\python_program\deepfake\DenseCLIP-master\data\ade\DeepfakeDataset'
dataset_type = 'DeepfakeDataset'

# 4. 数据预处理
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='ForceBinaryLabels', threshold=127),

    # 1. Resize 目标设为你的原图尺寸 299x299 (这步主要为了确保所有输入绝对统一，不会发生真正的放大/缩小插值)
    dict(type='Resize', img_scale=(299, 299), keep_ratio=True),

    dict(type='RandomFlip', prob=0.5),
    dict(type='Normalize', **img_norm_cfg),

    # 2. 核心：用 Pad 把 299 补齐到模型需要的 320，保持像素 1:1 无损！
    dict(type='Pad', size=(320, 320), pad_val=0, seg_pad_val=255),

    dict(type='DefaultFormatBundle'),
    dict(type='Collect',
         keys=['img', 'gt_semantic_seg'],
         meta_keys=['filename', 'ori_shape', 'img_shape', 'pad_shape',
                    'scale_factor', 'flip', 'flip_direction', 'text_prompts']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(299, 299), # 注意这里改为 299
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            # 测试时同样 Pad 到 320
            dict(type='Pad', size=(320, 320), pad_val=0, seg_pad_val=255),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect',
                 keys=['img'],
                 meta_keys=['filename', 'ori_shape', 'img_shape', 'pad_shape',
                            'scale_factor', 'flip', 'flip_direction', 'text_prompts']),
        ])
]

# 5. 配置 DataLoaders
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/training',
        ann_dir='annotations/training',
        text_json_path=data_root + '/text_infos/train_text.json',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/validation',
        ann_dir='annotations/validation',
        text_json_path=data_root + '/text_infos/val_text.json',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/validation',
        ann_dir='annotations/validation',
        text_json_path=data_root + '/text_infos/val_text.json',
        pipeline=test_pipeline)
)

# 6. 模型配置
model = dict(
    type='DenseCLIP',
    pretrained=r'F:\python_program\deepfake\DenseCLIP-master\pretrained\RN50.pt',
    class_names=['real', 'fake'],
    text_head=False,
    context_length=5,
    backbone=dict(
        type='CLIPResNetWithAttention',
        layers=[3, 4, 6, 3],
        style='pytorch',
        input_resolution=320),

    text_encoder=dict(
        type='CLIPTextContextEncoder',
        context_length=77,
        style='pytorch'),
    context_decoder=dict(
        type='ContextDecoder',
        context_length=16,
        transformer_width=256,
        transformer_heads=4,
        transformer_layers=6,
        visual_dim=1024,
        dropout=0.1,
        outdim=512,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2050],
        out_channels=256,
        num_outs=4),
    decode_head=dict(
        type='FPNHead',
        in_channels=[256, 256, 256, 256],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=256,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,

        # --- 修正点 1: ignore_index 应该放在这里 (默认即为255，显式写出更清晰) ---
        ignore_index=255,

        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, class_weight=[0.4, 1.0]),
            dict(type='DiceLoss', loss_weight=1.5) # Dice Loss 专注于重叠度
        ]
    ),
    identity_head=dict(
        type='IdentityHead',
        in_channels=2,   # 👈 增加这行 (匹配 score_map 的 2 个类)
        channels=2,      # 👈 增加这行
        num_classes=2,
        norm_cfg=dict(type='BN', requires_grad=True),

        # --- 修正点 3: 同上 ---
        ignore_index=255,

        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=0.2,
            class_weight=[0.4, 1.0] # 这里也需要加权
        )
    ),
    test_cfg=dict(mode='whole')
)