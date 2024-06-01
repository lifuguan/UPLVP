'''
Author: your name
Date: 2022-03-18 15:21:57
LastEditTime: 2022-04-22 12:38:39
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /research_workspace/configs/det/_base_/datasets/selfsup_detection.py
'''

# dataset settings
data_source = 'CustomImageNet'
dataset_type = 'SelfMaskDataset'
scales = [320, 336, 352, 368, 400, 416, 432, 448, 464, 480]
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_pipeline = [
    dict(type='RandomResizedCrop', size=1280, scale=(0.2, 1.)),
    dict(type='RandomGrayscale', p=0.2),
    dict(
        type='ColorJitter',
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.4),
    # dict(type='RandomHorizontalFlip'),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]

format_pipeline = [
    dict(type='CustomDefaultFormatBundle'),
    dict(type="CustomCollect", keys=['img'])
]

# prefetch
prefetch = False

# dataset summary
data = dict(
    imgs_per_gpu=4,  # total 32*8
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_source=dict(
            type=data_source,
            data_prefix='data/imagenet/train2017',
            ann_file=None,
        ),
        pipeline=train_pipeline,
        format_pipeline=format_pipeline,
        prefetch=prefetch),
    val=dict(
        type=dataset_type,
        data_source=dict(
            type=data_source,
            data_prefix='data/imagenet/train2017',
            ann_file=None,
        ),
        pipeline=train_pipeline,
        format_pipeline=format_pipeline,
        prefetch=prefetch))
