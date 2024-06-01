'''
Author: your name
Date: 2021-12-28 10:22:23
LastEditTime: 2024-01-09 09:01:29
LastEditors: lifuguan 1002732355@qq.com
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /research_workspace/configs/det/knet/knet_s3_r50_fpn_1x_coco.py
'''


_base_ = [
    '../_base_/schedules/schedule_8x.py',
    '../_base_/models/knet_s3_r50_fpn_ctw1500.py',
    '../_base_/datasets/ctw1500_instance.py',
    '../_base_/default_runtime.py'
]

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='knet_exp',
                entity='lifuguan',
                name='upknet_ctw1500'))
    ])

load_from = 'work_dirs/upknet_feature_coco_pretrain_labeled_prompt_ann/epoch_12.pth'



