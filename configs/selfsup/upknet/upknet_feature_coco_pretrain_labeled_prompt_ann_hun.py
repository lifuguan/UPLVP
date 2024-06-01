'''
Author: your name
Date: 2021-12-28 10:22:23
LastEditTime: 2024-01-03 14:32:54
LastEditors: lifuguan 1002732355@qq.com
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /research_workspace/configs/det/knet/knet_s3_r50_fpn_1x_coco.py
'''
custom_imports = dict(
    imports=[
        'custommd.datasets.coco_pseudo',
        'custommd.pipelines.transforms',
        'custommd.models.detectors.upknet_feature_hun',
        'custommd.models.kernel_updator',
        'custommd.models.knet.kernel_head',
        'custommd.models.knet.kernel_iter_head_hun',
        'custommd.models.knet.kernel_update_head',
        'custommd.models.knet.semantic_fpn_wrapper',
        'custommd.models.knet.mask_hungarian_assigner',
        'custommd.models.knet.mask_hungarian_direct_assigner',
        'custommd.models.knet.set_epoch_info_hook'
    ],
    allow_failed_imports=False)

custom_hooks = [dict(type='SetEpochInfoHook')]

_base_ = [
    '../_base_/schedules/schedule_1x.py',
    '../_base_/models/upknet_s3_r50_fpn_test_ann.py',
    '../_base_/datasets/coco_instance_labeled_pretrain_ann.py',
    '../_base_/upknet_runtime.py'
]