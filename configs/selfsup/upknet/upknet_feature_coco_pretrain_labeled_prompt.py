'''
Author: your name
Date: 2021-12-28 10:22:23
LastEditTime: 2022-05-28 00:23:14
LastEditors: lifuguan lifugan_10027@outlook.com
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /research_workspace/configs/det/knet/knet_s3_r50_fpn_1x_coco.py
'''
custom_imports = dict(
    imports=[
        'custommd.datasets.coco_pseudo',
        'custommd.pipelines.transforms',
        'custommd.models.detectors.upknet_feature',
        'custommd.models.kernel_updator',
        'custommd.models.knet.kernel_head',
        'custommd.models.knet.kernel_iter_head',
        'custommd.models.knet.kernel_update_head',
        'custommd.models.knet.semantic_fpn_wrapper',
        'custommd.models.knet.mask_hungarian_assigner',
    ],
    allow_failed_imports=False)

_base_ = [
    '../_base_/schedules/schedule_1x.py',
    '../_base_/models/upknet_s3_r50_fpn_test.py',
    '../_base_/datasets/coco_instance_labeled_pretrain.py',
    '../_base_/upknet_runtime.py'
]