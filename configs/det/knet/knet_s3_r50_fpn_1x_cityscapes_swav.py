'''
Author: your name
Date: 2021-12-28 10:22:23
LastEditTime: 2022-07-18 21:08:26
LastEditors: lifuguan lifugan_10027@outlook.com
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /research_workspace/configs/det/knet/knet_s3_r50_fpn_1x_coco.py
'''


_base_ = [
    '../_base_/schedules/schedule_2x.py',
    '../_base_/models/knet_s3_r50_fpn_cityscapes.py',
    '../_base_/datasets/cityscapes_instance.py',
    '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint='model_zoo/swav_800ep_pretrain.pth.tar')
    )
)

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
                name='knet_cityscapes_swav'))
    ])
