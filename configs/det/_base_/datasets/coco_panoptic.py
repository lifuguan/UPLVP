'''
Author: your name
Date: 2022-03-04 13:05:28
LastEditTime: 2022-07-21 03:17:35
LastEditors: lifuguan lifugan_10027@outlook.com
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /research_workspace/configs/det/_base_/datasets/coco_panoptic.py
'''
# dataset settings
dataset_type = 'CocoPanopticDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadPanopticAnnotations',
        with_bbox=True,
        with_mask=True,
        with_seg=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_semantic_seg']),
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
data = dict(
    samples_per_gpu=12,
    workers_per_gpu=12,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/panoptic_train2017.1@10.0.json',
        img_prefix=data_root + 'train2017/',
        seg_prefix=data_root + 'annotations/panoptic_train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/panoptic_val2017.json',
        img_prefix=data_root + 'val2017/',
        seg_prefix=data_root + 'annotations/panoptic_val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/panoptic_val2017.json',
        img_prefix=data_root + 'val2017/',
        seg_prefix=data_root + 'annotations/panoptic_val2017/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric=['pq'])
