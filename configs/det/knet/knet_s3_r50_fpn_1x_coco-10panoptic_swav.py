'''
Author: lifuguan lifugan_10027@outlook.com
Date: 2022-07-05 01:43:04
LastEditors: lifuguan lifugan_10027@outlook.com
LastEditTime: 2022-07-21 08:31:53
FilePath: /research_workspace/configs/det/knet/knet_s3_r50_fpn_1x_coco-panoptic.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
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

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint='model_zoo/swav_800ep_pretrain.pth.tar')
    )
)
data_root = 'data/coco/'

data = dict(
    train=dict(
        ann_file=data_root+'annotations/panoptic_train2017.1@10.0.json'
    )
)
