'''
Author: lifuguan 1002732355@qq.com
Date: 2023-01-28 04:47:55
LastEditors: lifuguan 1002732355@qq.com
LastEditTime: 2024-01-12 06:43:29
FilePath: /research_workspace/configs/det/_base_/datasets/ctw1500_instance.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline_netease = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]

test_pipeline_netease = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1600, 1600),
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
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train = dict(
        type=dataset_type,
        classes=classes,
        ann_file=f'{data_root}/instances_training.json',
        img_prefix=f'{data_root}/imgs',
        pipeline=train_pipeline_netease),
    val = dict(
        type=dataset_type,
        classes=classes,
        ann_file=f'{data_root}/instances_test.json',
        img_prefix=f'{data_root}/imgs',
        pipeline=test_pipeline_netease),
    test = dict(
        classes=classes,
        type=dataset_type,
        ann_file=f'{data_root}/instances_test.json',
        img_prefix=f'{data_root}/imgs',
        pipeline=test_pipeline_netease)
)

# ref: configs/maskrcnn/mask_rcnn_r50_fpn_160e_ctw1500.py
evaluation = dict(metric=['segm'], interval=1, save_best='segm_mAP')
