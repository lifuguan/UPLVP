'''
Author: lifuguan 1002732355@qq.com
Date: 2024-01-03 03:10:30
LastEditors: lifuguan 1002732355@qq.com
LastEditTime: 2024-01-03 04:59:22
FilePath: /research_workspace/custommd/models/detectors/get_text_feature.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# load clip model

import torch
import clip
from .get_categories import get_categories

def load_clip_model(model_name="ViT-L/14@336px"):
    # "ViT-L/14@336px" # the big model that OpenSeg uses

    print("Loading CLIP {} model...".format(model_name))
    clip_pretrained, _ = clip.load(model_name, device='cuda', jit=False)
    print("Finish loading")
    return clip_pretrained


# Openscene: build text features using clip
def extract_clip_feature(categories_names, clip_pretrained):
    
    if isinstance(categories_names, str):
        lines = categories_names.split(',')
    elif isinstance(categories_names, list):
        lines = categories_names
    else:
        raise NotImplementedError

    labels = []
    for line in lines:
        label = line
        labels.append(label)
    text = clip.tokenize(labels)
    text = text.cuda()
    text_features = clip_pretrained.encode_text(text)

    return text_features


def get_text_feature():
    with torch.no_grad():
            clip_model = load_clip_model()
            categories_names = get_categories('coco')
            text_features = extract_clip_feature(categories_names, clip_model)
            # normalize dim = C(768)
            text_features= text_features / text_features.norm(dim=-1, keepdim=True)
            # transpose [C, 768] -> [768, C]
            text_features = text_features.t().float()

            return text_features