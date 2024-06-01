
_base_ = '../_base_/default_runtime.py'
# dataset settings

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# In mstrain 3x config, img_scale=[(1333, 640), (1333, 800)],
# multiscale_mode='range'
train_pipeline = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 800)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
dataset_type = 'CocoDataset'
data_root = 'data/ctw1500'
classes=('text',)
# Use RepeatDataset to speed up training
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train = dict(
        type=dataset_type,
        classes=classes,
        ann_file=f'{data_root}/instances_training.json',
        img_prefix=f'{data_root}/imgs',
        pipeline=train_pipeline),
    val = dict(
        type=dataset_type,
        classes=classes,
        ann_file=f'{data_root}/instances_test.json',
        img_prefix=f'{data_root}/imgs',
        pipeline=test_pipeline),
    test = dict(
        classes=classes,
        type=dataset_type,
        ann_file=f'{data_root}/instances_test.json',
        img_prefix=f'{data_root}/imgs',
        pipeline=test_pipeline)
)

evaluation = dict(interval=10, metric=['segm'])

# optimizer
# this is different from the original 1x schedule that use SGD
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    weight_decay=0.05,
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.25)}))
optimizer_config = dict(grad_clip=dict(max_norm=1, norm_type=2))

# learning policy
# Experiments show that using step=[9, 11] has higher performance
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[9, 11])
runner = dict(type='EpochBasedRunner', max_epochs=120)
