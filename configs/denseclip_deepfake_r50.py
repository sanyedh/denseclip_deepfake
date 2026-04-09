USE_DETAILED_TEXT = False # 1. 详细文本
USE_ADAPTER = False # 2. 伪造痕迹适配器消融
USE_CROSS_ATTN = False     # 3. 跨模态注意力消融

# 1. 引用基础配置
_base_ = [
    '_base_/default_runtime.py',
    '_base_/schedules/schedule_18k.py'
]

# 2. 注册自定义 Dataset 模块
custom_imports = dict(imports=['mmseg_custom.datasets.deepfake'], allow_failed_imports=False)

# 3. 定义全局数据路径 (指向四个类别的父目录)
data_root = 'F:\\python_program\deepfake\\DenseCLIP-master\data\\ade\\DeepfakeDataset'
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
        use_detailed_text=USE_DETAILED_TEXT,
        dataset_split='training',  # 传入标记，告诉 Dataset 扫描 training 文件夹
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        dataset_split='validation', # 传入标记，告诉 Dataset 扫描 validation 文件夹
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        dataset_split='validation',
        pipeline=test_pipeline)
)

# 6. 模型配置
model = dict(
    type='DenseCLIP',
    use_adapter=USE_ADAPTER,
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
        type='CrossAttnTextGuidedFPNHead',
        use_cross_attn=USE_CROSS_ATTN,
        in_channels=[256, 256, 256, 256],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=256,
        text_channels=1024,
        num_heads=8,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='GN', num_groups=32,requires_grad=True),
        align_corners=False,
        ignore_index=255,
        loss_decode=[
            # 【修改 1】：引入 Focal Loss 替换 CrossEntropy
            dict(
                type='CustomFocalLoss',
                use_sigmoid=False,
                gamma=4.0,  # gamma 越大，对困难样本(如NT)的关注度越高，通常设为 2.0
                alpha=0.75,  # 平衡正负样本，通常设为 0.25
                loss_weight=1.0
            ),
            dict(type='DiceLoss', loss_weight=1.0, class_weight=[0.3, 0.7])
        ]
    ),

    identity_head=dict(
        type='IdentityHead',
        in_channels=2,
        channels=2,
        num_classes=2,
        norm_cfg=dict(type='GN', num_groups=2,requires_grad=True),
        ignore_index=255,

        loss_decode=dict(
            # 【修改 2】：同样将 IdentityHead 替换为 Focal Loss
            type='CustomFocalLoss',
            use_sigmoid=False,
            gamma=4.0,
            alpha=0.75,
            loss_weight=0.5
        )
    ),
    test_cfg=dict(mode='whole')
)