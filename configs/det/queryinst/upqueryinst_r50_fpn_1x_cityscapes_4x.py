'''
Author: your name
Date: 2022-01-01 11:37:32
LastEditTime: 2024-01-12 06:36:55
LastEditors: lifuguan 1002732355@qq.com
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /research_workspace/configs/det/knet/queryinst_r50_fpn_1x_coco.py
'''

_base_ = ['./queryinst_r50_fpn_1x_cityscapes_scratch_4x.py']

load_from = "work_dirs/upqueryinst_r50_fpn_1x_coco_pretrain_moco_labeled_prompt/epoch_12.pth"