'''
Author: lifuguan lifugan_10027@outlook.com
Date: 2022-05-08 12:06:13
LastEditors: lifuguan lifugan_10027@outlook.com
LastEditTime: 2022-07-19 16:12:33
FilePath: /research_workspace/configs/det/_base_/schedules/schedule_1x.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# optimizer
# this is different from the original 1x schedule that use SGD
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
    step=[16, 22])

runner = dict(type='EpochBasedRunner', max_epochs=24)
