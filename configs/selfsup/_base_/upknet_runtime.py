'''
Author: your name
Date: 2022-03-25 18:25:17
LastEditTime: 2022-05-16 23:40:23
LastEditors: lifuguan lifugan_10027@outlook.com
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /research_workspace/configs/det/_base_/selfsup_default_runtime.py
'''

# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        # dict(
        #     type='WandbLoggerHook',
        #     init_kwargs=dict(
        #         project='pretrain',
        #         entity='lifuguan',
        #         name='upknet_test'))
    ])
# yapf:enable

# runtime settings
dist_params = dict(backend='nccl')
cudnn_benchmark = True
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 12)]
persistent_workers = True
checkpoint_config = dict(interval=1)
evaluation = dict(interval=12, metric=['proposal'])
# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'
