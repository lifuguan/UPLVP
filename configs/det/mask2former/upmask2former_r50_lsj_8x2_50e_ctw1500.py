'''
Author: lifuguan lifugan_10027@outlook.com
Date: 2022-07-18 02:40:38
LastEditors: lifuguan 1002732355@qq.com
LastEditTime: 2024-01-12 03:11:16
FilePath: /research_workspace/configs/det/mask2former/upmask2former_r50_lsj_8x2_50e_coco.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
_base_ = ['./mask2former_r50_lsj_8x2_50e_ctw1500_scratch.py']

# ref: configs/maskrcnn/mask_rcnn_r50_fpn_160e_ctw1500.py
evaluation = dict(metric=['segm'], interval=1, save_best='segm_mAP')


log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook', by_epoch=False),
        
    ])

lr_config = dict(
    policy='step',
    gamma=0.1,
    step=[12, 18],
    warmup='linear',
    warmup_ratio=1.0,  # no warmup
    warmup_iters=10)

runner = dict(type='EpochBasedRunner', max_epochs=20) 

load_from = 'work_dirs/upmask2former_r50_lsj_8x2_50e_coco_prompt_ann/epoch_12.pth'
