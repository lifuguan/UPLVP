'''
Author: your name
Date: 2021-12-28 10:22:23
LastEditTime: 2022-06-19 12:29:06
LastEditors: lifuguan lifugan_10027@outlook.com
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /research_workspace/configs/det/knet/knet_s3_r50_fpn_1x_coco.py
'''


_base_ = [
    '../_base_/schedules/schedule_1x.py',
    '../_base_/datasets/coco_instance_pretrain.py',
    '../_base_/upknet_runtime.py'
]
custom_imports = dict(
    imports=[
        'mmseg.models',
        'custommd.datasets.coco_pseudo',
        'custommd.pipelines.transforms',
        'custommd.models.detectors.upencoder_decoder',
        'custommd.models.knet.iter_decode_head',
    ],
    allow_failed_imports=False)

# model settings
num_stages = 3
conv_kernel_size = 1
model = dict(
    type='UpEncoderDecoder',
    backbone=dict(
        type='ResNet',        
        depth=50,
        num_stages=4,
        out_indices=(0,1,2,3), # 只输出最后一层
        frozen_stages=4,
        norm_cfg=dict(type='SyncBN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(
            type='Pretrained', checkpoint='/disk1/lihao/model_zoo/densecl_resnet50.pth')), # torchvision://resnet50
    decode_head=dict(
        type='UpIterativeDecodeHead',
        num_stages=num_stages,
        kernel_update_head=[
            dict(
                type='KernelUpdateHead',
                num_classes=150,
                num_ffn_fcs=2,
                num_heads=8,
                num_mask_fcs=1,
                feedforward_channels=2048,
                in_channels=512,
                out_channels=512,
                dropout=0.0,
                conv_kernel_size=conv_kernel_size,
                ffn_act_cfg=dict(type='ReLU', inplace=True),
                with_ffn=True,
                feat_transform_cfg=dict(
                    conv_cfg=dict(type='Conv2d'), act_cfg=None),
                kernel_updator_cfg=dict(
                    type='KernelUpdator',
                    in_channels=256,
                    feat_channels=256,
                    out_channels=256,
                    act_cfg=dict(type='ReLU', inplace=True),
                    norm_cfg=dict(type='LN'))) for _ in range(num_stages)
        ],
        kernel_generate_head=dict(
            type='FCNHead',
            in_channels=2048,
            in_index=3,
            channels=512,
            num_convs=2,
            concat_input=True,
            dropout_ratio=0.1,
            num_classes=150,
            norm_cfg= dict(type='BN', requires_grad=True),
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg= dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))