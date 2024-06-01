'''
Author: your name
Date: 2022-04-12 19:26:09
LastEditTime: 2022-04-26 12:06:39
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /research_workspace/configs/selfsup/_base_/schedules/schedule_1x.py
'''
# optimizer
# this is different from the original 1x schedule that use SGD
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    weight_decay=0.05,)

optimizer_config = dict(grad_clip=dict(max_norm=1, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
