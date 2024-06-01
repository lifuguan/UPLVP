'''
Author: lifuguan lifugan_10027@outlook.com
Date: 2022-07-26 14:46:43
LastEditors: lifuguan lifugan_10027@outlook.com
LastEditTime: 2022-07-26 18:40:04
FilePath: /research_workspace/configs/det/_base_/schedules/schedule_5x_ocr.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# optimizer
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    weight_decay=0.05,
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.25)}))
optimizer_config = dict(grad_clip=dict(max_norm=4, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=0.001,
    step=[60, 72])
# running settings
runner = dict(type='EpochBasedRunner', max_epochs=80)