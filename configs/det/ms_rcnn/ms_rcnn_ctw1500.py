_base_ = '../mask_rcnn/mask_rcnn_r50_fpn_ctw1500_instance.py'
model = dict(
    type='MaskScoringRCNN',
    roi_head=dict(
        type='MaskScoringRoIHead',
        mask_iou_head=dict(
            type='MaskIoUHead',
            num_convs=4,
            num_fcs=2,
            roi_feat_size=14,
            in_channels=256,
            conv_out_channels=256,
            fc_out_channels=1024,
            num_classes=1)),
    # model training and testing settings
    train_cfg=dict(rcnn=dict(mask_thr_binary=0.5)))

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='knet_exp',
                entity='lifuguan',
                name='msrcnn_ctw1500'))
    ])