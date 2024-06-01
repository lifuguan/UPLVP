'''
Author: lifuguan lifugan_10027@outlook.com
Date: 2022-07-18 02:40:38
LastEditors: lifuguan 1002732355@qq.com
LastEditTime: 2024-01-10 12:30:33
FilePath: /research_workspace/configs/det/mask2former/upmask2former_r50_lsj_8x2_50e_coco.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
_base_ = ['./mask2former_r50_lsj_8x2_50e_cityscapes_scratch_4x.py']


load_from = 'work_dirs/upmask2former_r50_lsj_8x2_50e_coco_prompt_ann/epoch_12.pth'
