'''
Author: your name
Date: 2021-12-28 10:22:23
LastEditTime: 2022-07-26 01:26:47
LastEditors: lifuguan lifugan_10027@outlook.com
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /research_workspace/configs/det/knet/knet_s3_r50_fpn_1x_coco.py
'''


_base_ = [
    '../_base_/schedules/schedule_1x.py',
    '../_base_/models/knet_s3_r50_fpn_cityscapes.py',
    '../_base_/datasets/cityscapes_instance.py',
    '../_base_/default_runtime.py'
]

data_root = 'data/cityscapes/'
data = dict(
    train=dict(
        dataset=dict(
            ann_file=data_root +
            'annotations/instancesonly_filtered_gtFine_train.1@50.0.json')
    )
)

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint='model_zoo/moco_v2.pth.tar')
        )
)
lr_config = dict(step=[32, 44])
runner = dict(max_epochs=48)