'''
Author: lifuguan
Date: 2022-03-18 17:49:55
Based: 
Description: 读取imagenet数据集
FilePath: /research_workspace/custommd/datasets/selfdet.py
'''

import torch
from mmcv.utils import build_from_cfg
from torchvision.transforms import Compose

from mmselfsup.datasets.base import BaseDataset
from mmselfsup.datasets.builder import DATASETS, PIPELINES, build_datasource

@DATASETS.register_module()
class SelfMaskDataset(BaseDataset):
    """
    The dataset outputs single views of an image.
    """

    def __init__(self, data_source, pipeline, format_pipeline, prefetch=False):

        self.data_source = build_datasource(data_source)

        detection_transform = [
            build_from_cfg(p, PIPELINES) for p in pipeline
        ]

        format_transform = [
            build_from_cfg(p, PIPELINES) for p in format_pipeline
        ]

        self.detection_transform = Compose(detection_transform) # 从backbone输入
        self.format_transform = Compose(format_transform) # 从backbone输入

        self.prefetch = prefetch


    def __getitem__(self, idx):
        img, filename = self.data_source.get_img(idx)
        img = self.detection_transform(img)
        c, w, h = img.shape
        if w<=16 or h<=16:
            return self[(idx+1)%len(self)]
        # the format of the dataset is same with COCO.
        target = {
            'ori_shape': torch.as_tensor([int(c), int(h), int(w)]), 
            'img_shape': torch.as_tensor([int(c), int(h), int(w)]),
            'filename': filename,
            'img': img
        } 

        target = self.format_transform(target)

        return target
        
    # TODO 需要编写results总结代码
    def evaluate(self, results, logger=None):
        return NotImplemented

