'''
Author: your name
Date: 2021-12-14 15:48:09
LastEditTime: 2021-12-30 12:34:01
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /research_workspace/configs/det/knet/knet_s3_r50_fpn_1x_ctw1500.py
'''
_base_ = [
    '../_base_/schedules/schedule_1x_ocr.py',
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/netease_instance.py',
    '../_base_/runtime_10e.py'
]
