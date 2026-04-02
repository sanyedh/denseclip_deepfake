# ----------------- 1. 基础设置 -----------------
dataset_type = 'DeepfakeDataset'  # 我们稍后定义这个类
data_root = r'F:\python_program\deepfake\DenseCLIP-master\data\ade\DeepfakeDataset'

# CLIP 专用的均值和方差 (保持不变)
IMG_MEAN = [v * 255 for v in [0.48145466, 0.4578275, 0.40821073]]
IMG_VAR = [v * 255 for v in [0.26862954, 0.26130258, 0.27577711]]
img_norm_cfg = dict(mean=IMG_MEAN, std=IMG_VAR, to_rgb=True)

crop_size = (512, 512)

# 修改为 (2类) - 对应 label 0 和 label 1
classes = ('real', 'fake')
palette = [[0, 0, 0], [255, 255, 255]] # 0显示黑，1显示白

# ----------------- 2. 训练管道 (Train Pipeline) -----------------
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='ForceBinaryLabels', threshold=10), # 放在 Resize 之前非常重要
    dict(type='Resize', img_scale=(512, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect',
         keys=['img', 'gt_semantic_seg'],
         # text_prompts 由 Dataset 类直接注入，这里只需收集
         meta_keys=('filename', 'ori_filename', 'ori_shape', 'img_shape',
                    'pad_shape', 'scale_factor', 'flip', 'flip_direction',
                    'img_norm_cfg', 'text_prompts'))
]

# ----------------- 3. 测试管道 (Test Pipeline) -----------------
test_pipeline = [
    dict(type='LoadImageFromFile'),
    # 测试时也需要加载文本
    dict(type='LoadDeepfakeText', json_path=data_root + '/texts/val_fftg.json'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect',
                 keys=['img'],
                 meta_keys=('filename', 'ori_filename', 'ori_shape', 'img_shape',
                            'pad_shape', 'scale_factor', 'flip', 'flip_direction',
                            'img_norm_cfg', 'text_prompts'))
        ])
]

# ----------------- 4. 数据集配置 -----------------
data = dict(
    samples_per_gpu=4,  # 根据显存调整
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/training',
        ann_dir='annotations/training',
        classes=classes,  # 显式传入类别
        palette=palette,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/validation',
        ann_dir='annotations/validation',
        classes=classes,
        palette=palette,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/validation',
        ann_dir='annotations/validation',
        classes=classes,
        palette=palette,
        pipeline=test_pipeline))