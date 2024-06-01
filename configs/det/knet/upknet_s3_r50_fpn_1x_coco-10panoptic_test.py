'''
Author: lifuguan lifugan_10027@outlook.com
Date: 2022-07-05 01:43:04
LastEditors: lifuguan lifugan_10027@outlook.com
LastEditTime: 2022-07-21 08:57:08
FilePath: /research_workspace/configs/det/knet/knet_s3_r50_fpn_1x_coco-panoptic.py
Description: test 使用有监督实例分割的模型在全景分割上finetune
'''
custom_imports = dict(
    imports=[
        'custommd.models.detectors.knet',
        'custommd.models.knet.kernel_head',
        'custommd.models.knet.kernel_iter_head',
        'custommd.models.knet.kernel_update_head',
        'custommd.models.knet.semantic_fpn_wrapper',
        'custommd.models.kernel_updator',
        'custommd.models.knet.mask_hungarian_assigner',
        # 'custommd.models.knet.mask_pseudo_sampler',
    ],
    allow_failed_imports=False)

_base_ = [
    '../_base_/schedules/schedule_1x.py',
    '../_base_/models/knet_s3_r50_fpn_panoptic.py',
    '../_base_/datasets/coco_panoptic.py',
    '../_base_/default_runtime.py'
]

data_root = 'data/coco/'

data = dict(
    train=dict(
        ann_file=data_root+'annotations/panoptic_train2017.1@10.0.json'
    )
)

load_from = 'work_dirs/knet_s3_r50_fpn_1x_cityscapes_densecl/epoch_12.pth'
