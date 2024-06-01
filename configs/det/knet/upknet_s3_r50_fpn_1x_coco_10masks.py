'''
Author: your name
Date: 2021-12-28 10:22:23
LastEditTime: 2022-01-26 17:06:13
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /research_workspace/configs/det/knet/knet_s3_r50_fpn_1x_coco.py
'''


_base_ = [
    '../_base_/schedules/schedule_1x.py',
    '../_base_/models/knet_s3_r50_fpn_netease.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/default_runtime.py'
]


evaluation = dict(interval=1)

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=50,
    warmup_ratio=0.001,
    step=[16, 22])

# lr_config = dict(
#     policy='CosineAnnealing',  # cosine learning policy
#     warmup='linear',
#     warmup_iters=1000,
#     warmup_ratio=0.001,
#     min_lr_ratio=1e-5)
runner = dict(type='EpochBasedRunner', max_epochs=24)

load_from = 'work_dirs/exp_densecl_12.pth'
