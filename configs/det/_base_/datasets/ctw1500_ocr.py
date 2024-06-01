'''
Author: lifuguan
Date: 2021-12-14 15:20:18
LastEditTime: 2021-12-21 09:54:31
LastEditors: Please set LastEditors
Description: 仿照mask-rcnn进行编写
FilePath: /research_workspace/configs/det/_base_/datasets/ctw1500_instance.py
'''

# pipline based on Mask-RCNN configs/_base_/det_pipelines/maskrcnn_pipeline.py

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline_ctw1500 = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='ScaleAspectJitter',
        img_scale=None,
        keep_ratio=False,
        resize_type='indep_sample_in_range',
        scale_range=(640, 2560)),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(
        type='RandomCropInstances',
        target_size=(640, 640),
        mask_type='union_all',
        instance_key='gt_masks'),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]

# for ctw1500
img_scale_ctw1500 = (1600, 1600)
test_pipeline_ctw1500 = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale_ctw1500,
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


# dataset based on ctw1500 
# ref: configs/_base_/det_datasets/ctw1500.py
# ref: configs/det/_base_/datasets/coco_instance.py

# dataset_type = 'CocoDataset'
dataset_type = 'IcdarDataset'
data_root = 'data/ctw1500'
classes=("text",)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train = dict(
        type=dataset_type,
        classes=classes,
        ann_file=f'{data_root}/instances_training.json',
        img_prefix=f'{data_root}/imgs',
        pipeline=train_pipeline_ctw1500),
    val = dict(
        classes=classes,
        type=dataset_type,
        ann_file=f'{data_root}/instances_test.json',
        img_prefix=f'{data_root}/imgs',
        pipeline=test_pipeline_ctw1500),
    test = dict(
        classes=classes,
        type=dataset_type,
        ann_file=f'{data_root}/instances_test.json',
        img_prefix=f'{data_root}/imgs',
        pipeline=test_pipeline_ctw1500)
)

# ref: configs/maskrcnn/mask_rcnn_r50_fpn_160e_ctw1500.py
evaluation = dict(interval=50, metric='hmean-iou')