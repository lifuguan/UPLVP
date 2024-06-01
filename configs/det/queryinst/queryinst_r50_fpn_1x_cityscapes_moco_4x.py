_base_ = ['./queryinst_r50_fpn_1x_cityscapes_scratch_4x.py']

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint='model_zoo/moco_v2.pth.tar')
    )
)