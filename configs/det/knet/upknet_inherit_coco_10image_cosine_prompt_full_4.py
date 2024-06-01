'''
Author: your name
Date: 2021-12-28 10:22:23
LastEditTime: 2022-01-26 17:06:13
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /research_workspace/configs/det/knet/knet_s3_r50_fpn_1x_coco.py
'''


_base_ = [
    '../_base_/schedules/schedule_1x.py',
    '../_base_/models/knet_s3_r50_fpn_netease.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/default_runtime.py'
]

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
    step=[36, 44])

# lr_config = dict(
#     policy='CosineAnnealing',  # cosine learning policy
#     warmup='linear',
#     warmup_iters=1000,
#     warmup_ratio=0.001,
#     min_lr_ratio=1e-5)
runner = dict(type='EpochBasedRunner', max_epochs=48)

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint='model_zoo/densecl_resnet50.pth')
    )
)

data = dict(
    samples_per_gpu=12,
    workers_per_gpu=12,
    train=dict(
        ann_file='data/coco/annotations/instances_train2017_sup10_seed1_labeled.json'
    )
)

load_from = 'work_dirs/upknet_cosine_coco_pretrain_labeled_prompt/epoch_4.pth'