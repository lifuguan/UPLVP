_base_ = ['./queryinst_r50_fpn_1x_cityscapes_scratch_4x.py']

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint='model_zoo/swav_800ep_pretrain.pth.tar')
    )
)