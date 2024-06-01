'''
Author: lifuguan 1002732355@qq.com
Date: 2022-10-28 12:48:36
LastEditors: lifuguan 1002732355@qq.com
LastEditTime: 2024-01-02 07:42:05
FilePath: /research_workspace/custommd/models/utils/__init__.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.utils import Registry, build_from_cfg

from .box_util import is_on_same_line, stitch_boxes_into_lines
from .check_argument import (equal_len, is_2dlist, is_3dlist, is_none_or_type,
                             is_type_list, valid_boundary)
from .collect_env import collect_env
from .data_convert_util import convert_annotations
from .fileio import list_from_file, list_to_file
from .img_util import drop_orientation, is_not_png
from .lmdb_util import lmdb_converter
from .logger import get_root_logger
from .model import revert_sync_batchnorm
from .string_util import StringStrip
from .dataset_util import get_categories

__all__ = [
    'Registry', 'build_from_cfg', 'get_root_logger', 'collect_env',
    'is_3dlist', 'is_type_list', 'is_none_or_type', 'equal_len', 'is_2dlist',
    'valid_boundary', 'lmdb_converter', 'drop_orientation',
    'convert_annotations', 'is_not_png', 'list_to_file', 'list_from_file',
    'is_on_same_line', 'stitch_boxes_into_lines', 'StringStrip',
    'revert_sync_batchnorm', 'get_categories'
]
