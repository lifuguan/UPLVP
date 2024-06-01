'''
Author: lifuguan lifugan_10027@outlook.com
Date: 2022-07-13 21:18:29
LastEditors: lifuguan lifugan_10027@outlook.com
LastEditTime: 2022-07-14 09:43:56
FilePath: /research_workspace/configs/det/_base_/datasets/cityscapes_instance.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# dataset settings
dataset_type = 'CityscapesDataset'
data_root = 'data/cityscapes/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='Resize', img_scale=[(2048, 800), (2048, 1024)], keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 1024),
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
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=8,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root +
            'annotations/instancesonly_filtered_gtFine_train.json',
            img_prefix=data_root + 'leftImg8bit/train/',
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        ann_file=data_root +
        'annotations/instancesonly_filtered_gtFine_val.json',
        img_prefix=data_root + 'leftImg8bit/val/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root +
        'annotations/instancesonly_filtered_gtFine_test.json',
        img_prefix=data_root + 'leftImg8bit/test/',
        pipeline=test_pipeline))
evaluation = dict(metric=['segm'], interval=1, save_best='segm_mAP')
