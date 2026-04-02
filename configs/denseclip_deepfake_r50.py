# 1. 引用基础配置
_base_ = [
    '_base_/default_runtime.py',
    '_base_/schedules/schedule_32k.py'
]

# 2. 注册自定义 Dataset 模块
custom_imports = dict(imports=['mmseg_custom.datasets.deepfake'], allow_failed_imports=False)

# 3. 定义数据路径
data_root = 'F:/python_program/deepfake/DenseCLIP-master/data/ade/DeepfakeDataset/NeuralTextures'
dataset_type = 'DeepfakeDataset'

# 4. 数据预处理
img_norm_cfg = dict(
    mean=[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255],
    std=[0.26862954 * 255, 0.26130258 * 255, 0.27577711 * 255],
    to_rgb=True
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='ForceBinaryLabels', threshold=10),
    dict(type='Resize', img_scale=(299, 299), ratio_range=(0.8, 1.2)),
    dict(type='RandomCrop', crop_size=(299, 299), cat_max_ratio=0.98),
    dict(type='RandomFlip', prob=0.5),

    # ---------------- 核心修改区 ----------------
    # 1. 移除或注释掉原有的强色彩畸变，防止破坏 Color Mismatch 伪影
    # dict(type='PhotoMetricDistortion'),

    # # 2. 引入 Deepfake 领域专属的降质增强 (50%概率触发)
    # dict(type='RandomJPEGCompression', quality_lower=60, quality_upper=95, p=0.5),
    # dict(type='RandomGaussianBlur', kernel_size_list=[3, 5], p=0.5),
    # # -------------------------------------------

    dict(type='Normalize', **img_norm_cfg),
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
        img_scale=(299, 299),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size=(320, 320), pad_val=0, seg_pad_val=255),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect',
                 keys=['img'],
                 meta_keys=['filename', 'ori_shape', 'img_shape', 'pad_shape','ori_filename',
                            'scale_factor', 'flip', 'flip_direction', 'text_prompts','img_norm_cfg']),
        ])
]

# 5. 配置 DataLoaders
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=6,
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
    text_head=True,
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
        type='CrossAttnTextGuidedFPNHead',  # <--- 修改为新注册的Head
        in_channels=[256, 256, 256, 256],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=256,
        text_channels=1024,   # 新增：定义文本维度
        num_heads=8,          # 新增：注意力头数
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        ignore_index=255,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, class_weight=[0.3, 0.7]),
            dict(type='DiceLoss', loss_weight=1.5,class_weight=[0.3, 0.7])
        ]
    ),

    identity_head=dict(
        type='IdentityHead',
        in_channels=2,
        channels=2,
        num_classes=2,
        norm_cfg=dict(type='BN', requires_grad=True),
        ignore_index=255,

        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.5, class_weight=[0.3, 0.7]
        )
    ),
    test_cfg=dict(mode='whole')
)

