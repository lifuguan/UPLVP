'''
Author: your name
Date: 2022-03-15 15:51:42
LastEditTime: 2022-07-15 01:41:33
LastEditors: lifuguan lifugan_10027@outlook.com
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /research_workspace/Untitled-1.py
'''
# %%

import torch
import re
dict = torch.load("../work_dirs/oursnet_s3_r50_fpn_1x_coco/epoch_12.pth")
# backbone = torch.load("../model_zoo/swav_800ep_pretrain.pth.tar")
backbone = torch.load("../model_zoo/moco_v2.pth.tar")
#%%
patt2='backbone'
pattern2 = re.compile(patt2)
key_list = list(dict['state_dict'].keys())
for older_val in key_list:
    if len(pattern2.findall(older_val)) != 0:
    # if len(pattern2.findall(older_val)) != 0:
        dict['state_dict'].pop(older_val)

backbone_list = list(backbone['state_dict'].keys())
for list_ in backbone_list:
    val = re.sub('module', 'backbone', list_)
    # print(list_)
    dict['state_dict'][val] = backbone['state_dict'].pop(list_)
print(dict['state_dict'].keys())
torch.save(dict, '../work_dirs/oursnet_moco_12.pth')

#%% MoCo
import torch
import re

backbone = torch.load("/disk1/lihao/model_zoo/moco_v2_800ep_pretrain.pth.tar")
key_list = list(backbone['state_dict'].keys())
#%%

for list_ in key_list:
    val = re.sub('module.encoder_q.', 'module.', list_)
    backbone['state_dict'][val] = backbone['state_dict'].pop(list_)
print(backbone['state_dict'].keys())
torch.save(backbone, '/disk1/lihao/model_zoo/moco_v2.pth.tar')




# %% 适用于K-Net的语义分割模型权重
import torch
import re
dict = torch.load("/disk2/lihao/research_workspace/work_dirs/epoch_12.pth")
print(dict['state_dict'].keys())
patt='neck'
pattern = re.compile(patt)
patt2='backbone'
pattern2 = re.compile(patt2)
patt3='rpn_head'
pattern3 = re.compile(patt3)
patt4='roi_head'
pattern4 = re.compile(patt4)
key_list = list(dict['state_dict'].keys())
for older_val in key_list:
    if len(pattern.findall(older_val)) != 0 or len(pattern2.findall(older_val)) != 0 or len(pattern3.findall(older_val)) != 0:
    # if len(pattern2.findall(older_val)) != 0:
        dict['state_dict'].pop(older_val)
    
    if len(pattern4.findall(older_val)) != 0:
        val = re.sub('roi_head.mask_head.', 'decode_head.kernel_update_head.', older_val)
        dict['state_dict'][val] = dict['state_dict'].pop(older_val)
print(dict['state_dict'].keys())
torch.save(dict, '/disk2/lihao/research_workspace/work_dirs/knet_semantic_12.pth')

# %%
import torch
import re
# dict = torch.load("../work_dirs/exp_12.pth")

dict = torch.load("../work_dirs/baseline/knet_s3_r50_fpn_1x_coco_10image_densecl/epoch_3.pth")
key_list = list(dict['state_dict'].keys())


# %%
