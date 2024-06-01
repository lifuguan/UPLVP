'''
Author: lifuguan 1002732355@qq.com
Date: 2024-01-03 14:30:20
LastEditors: lifuguan 1002732355@qq.com
LastEditTime: 2024-01-03 14:30:45
FilePath: /research_workspace/custommd/models/detectors/get_epoch_hookl.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

from mmcv.parallel import is_module_wrapper
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class SetEpochInfoHook(Hook):
    """Set runner's epoch information to the model."""

    def before_train_epoch(self, runner):
        epoch = runner.epoch
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        model.set_epoch(epoch)
