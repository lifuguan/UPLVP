'''
Author: your name
Date: 2021-12-28 10:22:23
LastEditTime: 2022-07-15 16:20:37
LastEditors: lifuguan lifugan_10027@outlook.com
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /research_workspace/configs/det/knet/knet_s3_r50_fpn_1x_coco.py
'''


_base_ = [
    '../_base_/schedules/schedule_1x.py',
    '../_base_/models/knet_s3_r50_fpn_netease.py',
    '../_base_/datasets/cityscapes_instance.py',
    '../_base_/default_runtime.py'
]
bs=8
num_classes=8

lr_config = dict(step=[8, 11])
runner = dict(max_epochs=12)

load_from = 'work_dirs/exp_moco_12.pth'