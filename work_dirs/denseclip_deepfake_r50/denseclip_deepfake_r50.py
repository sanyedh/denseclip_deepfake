log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook')
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
find_unused_parameters = True
optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.0001)
optimizer_config = dict()
lr_config = dict(
    policy='poly',
    power=0.9,
    min_lr=1e-06,
    by_epoch=False,
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06)
runner = dict(type='IterBasedRunner', max_iters=80000)
checkpoint_config = dict(by_epoch=False, interval=8000)
evaluation = dict(interval=4000, metric='mIoU', save_best='mIoU')
custom_imports = dict(
    imports=['mmseg_custom.datasets.deepfake'], allow_failed_imports=False)
data_root = 'F:\python_program\deepfake\DenseCLIP-master\data\ade\DeepfakeDataset\Deepfakes'
dataset_type = 'DeepfakeDataset'
img_norm_cfg = dict(
    mean=[122.7709383, 116.7460125, 104.09373615000001],
    std=[68.5005327, 66.6321579, 70.32316304999999],
    to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='ForceBinaryLabels', threshold=10),
    dict(type='Resize', img_scale=(299, 299), ratio_range=(0.8, 1.2)),
    dict(type='RandomCrop', crop_size=(299, 299), cat_max_ratio=0.98),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[122.7709383, 116.7460125, 104.09373615000001],
        std=[68.5005327, 66.6321579, 70.32316304999999],
        to_rgb=True),
    dict(type='Pad', size=(320, 320), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_semantic_seg'],
        meta_keys=[
            'filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor',
            'flip', 'flip_direction', 'text_prompts'
        ])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(299, 299),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(
                type='Normalize',
                mean=[122.7709383, 116.7460125, 104.09373615000001],
                std=[68.5005327, 66.6321579, 70.32316304999999],
                to_rgb=True),
            dict(type='Pad', size=(320, 320), pad_val=0, seg_pad_val=255),
            dict(type='ImageToTensor', keys=['img']),
            dict(
                type='Collect',
                keys=['img'],
                meta_keys=[
                    'filename', 'ori_shape', 'img_shape', 'pad_shape',
                    'scale_factor', 'flip', 'flip_direction', 'text_prompts'
                ])
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type='DeepfakeDataset',
        data_root=
        'F:\python_program\deepfake\DenseCLIP-master\data\ade\DeepfakeDataset\Deepfakes',
        img_dir='images/training',
        ann_dir='annotations/training',
        text_json_path=
        'F:\python_program\deepfake\DenseCLIP-master\data\ade\DeepfakeDataset\Deepfakes/text_infos/train_text.json',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', reduce_zero_label=False),
            dict(type='ForceBinaryLabels', threshold=10),
            dict(type='Resize', img_scale=(299, 299), ratio_range=(0.8, 1.2)),
            dict(type='RandomCrop', crop_size=(299, 299), cat_max_ratio=0.98),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PhotoMetricDistortion'),
            dict(
                type='Normalize',
                mean=[122.7709383, 116.7460125, 104.09373615000001],
                std=[68.5005327, 66.6321579, 70.32316304999999],
                to_rgb=True),
            dict(type='Pad', size=(320, 320), pad_val=0, seg_pad_val=255),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=['img', 'gt_semantic_seg'],
                meta_keys=[
                    'filename', 'ori_shape', 'img_shape', 'pad_shape',
                    'scale_factor', 'flip', 'flip_direction', 'text_prompts'
                ])
        ]),
    val=dict(
        type='DeepfakeDataset',
        data_root=
        'F:\python_program\deepfake\DenseCLIP-master\data\ade\DeepfakeDataset\Deepfakes',
        img_dir='images/validation',
        ann_dir='annotations/validation',
        text_json_path=
        'F:\python_program\deepfake\DenseCLIP-master\data\ade\DeepfakeDataset\Deepfakes/text_infos/val_text.json',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(299, 299),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(
                        type='Normalize',
                        mean=[122.7709383, 116.7460125, 104.09373615000001],
                        std=[68.5005327, 66.6321579, 70.32316304999999],
                        to_rgb=True),
                    dict(
                        type='Pad',
                        size=(320, 320),
                        pad_val=0,
                        seg_pad_val=255),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(
                        type='Collect',
                        keys=['img'],
                        meta_keys=[
                            'filename', 'ori_shape', 'img_shape', 'pad_shape',
                            'scale_factor', 'flip', 'flip_direction',
                            'text_prompts'
                        ])
                ])
        ]),
    test=dict(
        type='DeepfakeDataset',
        data_root=
        'F:\python_program\deepfake\DenseCLIP-master\data\ade\DeepfakeDataset\Deepfakes',
        img_dir='images/validation',
        ann_dir='annotations/validation',
        text_json_path=
        'F:\python_program\deepfake\DenseCLIP-master\data\ade\DeepfakeDataset\Deepfakes/text_infos/val_text.json',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(299, 299),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(
                        type='Normalize',
                        mean=[122.7709383, 116.7460125, 104.09373615000001],
                        std=[68.5005327, 66.6321579, 70.32316304999999],
                        to_rgb=True),
                    dict(
                        type='Pad',
                        size=(320, 320),
                        pad_val=0,
                        seg_pad_val=255),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(
                        type='Collect',
                        keys=['img'],
                        meta_keys=[
                            'filename', 'ori_shape', 'img_shape', 'pad_shape',
                            'scale_factor', 'flip', 'flip_direction',
                            'text_prompts'
                        ])
                ])
        ]))
model = dict(
    type='DenseCLIP',
    pretrained='F:\python_program\deepfake\DenseCLIP-master\pretrained\RN50.pt',
    class_names=['real', 'fake'],
    text_head=True,
    context_length=5,
    backbone=dict(
        type='CLIPResNetWithAttention',
        layers=[3, 4, 6, 3],
        style='pytorch',
        input_resolution=320),
    text_encoder=dict(
        type='CLIPTextContextEncoder', context_length=77, style='pytorch'),
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
        type='TextGuidedFPNHead',
        in_channels=[256, 256, 256, 256],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=256,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        ignore_index=255,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0,
                class_weight=[0.5, 0.5]),
            dict(type='DiceLoss', loss_weight=0.5)
        ]),
    identity_head=dict(
        type='IdentityHead',
        in_channels=2,
        channels=2,
        num_classes=2,
        norm_cfg=dict(type='BN', requires_grad=True),
        ignore_index=255,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=[0.1, 0.9])),
    test_cfg=dict(mode='whole'))
work_dir = './work_dirs\denseclip_deepfake_r50'
gpu_ids = range(0, 1)
