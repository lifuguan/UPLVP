'''
Author: your name
Date: 2021-12-28 10:22:23
LastEditTime: 2022-04-14 11:12:30
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /research_workspace/configs/det/knet/knet_s3_r50_fpn_1x_coco.py
'''

_base_ = [
    '../_base_/schedules/schedule_1x.py',
    '../_base_/models/oursnet_s3_r50_fpn_test.py',
    '../_base_/datasets/coco_instance_labeled_pretrain.py',
    '../_base_/upknet_runtime.py'
]
