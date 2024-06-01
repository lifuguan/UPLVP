'''
Author: lifuguan
Date: 2022-03-01 16:48:51
LastEditTime: 2022-03-01 17:22:15
LastEditors: Please set LastEditors
Description: ReCTS instance segmentation
FilePath: /research_workspace/configs/det/_base_/datasets/rects_instance.py
'''
dataset_type = 'CocoDataset'
data_root = 'data/ReCTS/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
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
classes=('text',)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        classes=classes,
        type=dataset_type,
        ann_file=data_root + 'annotations/rects_train.json',
        img_prefix=data_root + 'ReCTS_train_images/',
        pipeline=train_pipeline),
    val=dict(
        classes=classes,
        type=dataset_type,
        ann_file=data_root + 'annotations/rects_val.json',
        img_prefix=data_root + 'ReCTS_test_images/',
        pipeline=test_pipeline),
    test=dict(
        classes=classes,
        type=dataset_type,
        ann_file=data_root + 'annotations/rects_val.json',
        img_prefix=data_root + 'ReCTS_test_images/',
        pipeline=test_pipeline))
# we do not evaluate bbox because K-Net does not predict bounding boxes
evaluation = dict(metric=['bbox'])