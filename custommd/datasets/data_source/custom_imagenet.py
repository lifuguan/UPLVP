'''
Author: your name
Date: 2022-03-30 17:13:07
LastEditTime: 2022-03-30 23:35:47
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /research_workspace/custommd/datasource/custom_imagenet.py
'''

# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
from PIL import Image

import numpy as np

import mmcv
from mmselfsup.datasets.builder import DATASOURCES
from mmselfsup.datasets.data_sources.base import BaseDataSource



def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def find_folders(root):
    """Find classes by folders under a root.

    Args:
        root (string): root directory of folders

    Returns:
        folder_to_idx (dict): the map from folder name to class idx
    """
    folders = [d for d in os.listdir(root) if osp.isdir(osp.join(root, d))]
    folders.sort()
    folder_to_idx = {folders[i]: i for i in range(len(folders))}
    return folder_to_idx


def get_samples(root, folder_to_idx, extensions):
    """Make dataset by walking all images under a root.

    Args:
        root (string): root directory of folders
        folder_to_idx (dict): the map from class name to class idx
        extensions (tuple): allowed extensions

    Returns:
        samples (list): a list of tuple where each element is (image, label)
    """
    samples = []
    root = osp.expanduser(root)
    for folder_name in sorted(list(folder_to_idx.keys())):
        _dir = osp.join(root, folder_name)
        for _, _, fns in sorted(os.walk(_dir)):
            for fn in sorted(fns):
                if has_file_allowed_extension(fn, extensions):
                    path = osp.join(folder_name, fn)
                    item = (path, folder_to_idx[folder_name])
                    samples.append(item)
    return samples


@DATASOURCES.register_module()
class CustomImageNet(BaseDataSource):
    """`ImageNet <http://www.image-net.org>`_ Dataset.

    This implementation is modified from
    https://github.com/pytorch/vision/blob/master/torchvision/datasets/imagenet.py
    """  # noqa: E501

    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif')

    def get_img(self, idx):
        """Get image by index.

        Args:
            idx (int): Index of data.

        Returns:
            Image: PIL Image format.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if self.data_infos[idx].get('img_prefix', None) is not None:
            if self.data_infos[idx]['img_prefix'] is not None:
                filename = osp.join(
                    self.data_infos[idx]['img_prefix'],
                    self.data_infos[idx]['img_info']['filename'])
            else:
                filename = self.data_infos[idx]['img_info']['filename']
            img_bytes = self.file_client.get(filename)
            img = mmcv.imfrombytes(img_bytes, flag=self.color_type)
        else:
            img = self.data_infos[idx]['img']

        img = img.astype(np.uint8)
        return Image.fromarray(img), filename

    def load_annotations(self):
        if self.ann_file is None:
            folder_to_idx = find_folders(self.data_prefix)
            samples = get_samples(
                self.data_prefix,
                folder_to_idx,
                extensions=self.IMG_EXTENSIONS)
            if len(samples) == 0:
                raise (RuntimeError('Found 0 files in subfolders of: '
                                    f'{self.data_prefix}. '
                                    'Supported extensions are: '
                                    f'{",".join(self.IMG_EXTENSIONS)}'))

            self.folder_to_idx = folder_to_idx
        elif isinstance(self.ann_file, str):
            with open(self.ann_file) as f:
                samples = [x.strip().rsplit(' ', 1) for x in f.readlines()]
        else:
            raise TypeError('ann_file must be a str or None')
        self.samples = samples

        data_infos = []
        for i, (filename, gt_label) in enumerate(self.samples):
            info = {'img_prefix': self.data_prefix}
            info['img_info'] = {'filename': filename}
            info['gt_label'] = np.array(gt_label, dtype=np.int64)
            info['idx'] = int(i)
            data_infos.append(info)
        return data_infos
