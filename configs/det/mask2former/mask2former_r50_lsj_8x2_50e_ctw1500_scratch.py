'''
Author: lifuguan lifugan_10027@outlook.com
Date: 2022-07-18 02:40:38
LastEditors: lifuguan lifugan_10027@outlook.com
LastEditTime: 2022-07-18 11:03:27
FilePath: /research_workspace/configs/det/mask2former/upmask2former_r50_lsj_8x2_50e_coco.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
_base_ = ['./mask2former_r50_lsj_8x2_50e_10image_scratch.py']

num_things_classes = 1
num_stuff_classes = 0
num_classes = num_things_classes + num_stuff_classes
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
    ),
    panoptic_head=dict(
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        loss_cls=dict(class_weight=[1.0] * num_classes + [0.1])),
    panoptic_fusion_head=dict(
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes),
    test_cfg=dict(panoptic_on=False))

# dataset settings
image_size = (1024, 1024)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
pad_cfg = dict(img=(128, 128, 128), masks=0, seg=255)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    # large scale jittering
    dict(
        type='Resize',
        img_scale=image_size,
        ratio_range=(0.1, 2.0),
        multiscale_mode='range',
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_size=image_size,
        crop_type='absolute',
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-5, 1e-5), by_mask=True),
    dict(type='Pad', size=image_size, pad_val=pad_cfg),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle', img_to_float=True),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
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
            dict(type='Pad', size_divisor=32, pad_val=pad_cfg),
            dict(type='Normalize', **img_norm_cfg),
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

# ref: configs/maskrcnn/mask_rcnn_r50_fpn_160e_ctw1500.py
evaluation = dict(metric=['segm'], interval=1, save_best='segm_mAP')

# learning policy
lr_config = dict(
    policy='step',
    gamma=0.1,
    step=[30, 36],
    warmup='linear',
    warmup_ratio=1.0,  # no warmup
    warmup_iters=10)

runner = dict(type='EpochBasedRunner', max_epochs=40)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook', by_epoch=False),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='oursnet',
                name='lifuguan'))
    ])

