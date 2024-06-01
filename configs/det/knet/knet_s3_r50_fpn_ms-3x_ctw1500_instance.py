'''
Author: your name
Date: 2021-12-22 16:57:10
LastEditTime: 2021-12-22 16:57:10
LastEditors: your name
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /research_workspace/configs/det/knet/knet_s3_r50_fpn_ms-3x_ctw1500_instance.py
'''
_base_ = [
    '../_base_/models/knet_s3_r50_fpn_netease.py',
    '../common/mstrain_3x_ctw1500_instance.py'
]
