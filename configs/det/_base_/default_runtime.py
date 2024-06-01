'''
Author: your name
Date: 2022-01-02 10:56:11
LastEditTime: 2022-07-14 02:22:30
LastEditors: lifuguan lifugan_10027@outlook.com
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /research_workspace/configs/det/_base_/default_runtime.py
'''

checkpoint_config = dict(interval=-1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        # dict(
        #     type='WandbLoggerHook',
        #     init_kwargs=dict(
        #         project='knet_exp',
        #         entity='knet',
        #         name='lifuguan'))
    ])
# yapf:enable
# custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
# workflow = [('train', 3), ('val', 1)]
workflow = [('train', 1)]
