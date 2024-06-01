'''
Author: your name
Date: 2021-12-20 16:47:41
LastEditTime: 2022-04-25 10:58:01
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /research_workspace/tools_det/demo.py
'''
import multiprocessing as mp
import pycocotools.mask as mask_util

from argparse import ArgumentParser
import glob
import json
import numpy as np

import torch.nn.functional as F
import torch

from mmdet.apis import init_detector
from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose
from tqdm import tqdm

from custommd.models.oursnet.utils import matrix_nms, center_of_mass


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--config', default='configs/selfsup/upknet/oursnet_s3_r50_fpn_1x_coco.py')
    parser.add_argument('--input', default='data/coco/train2017/')
    parser.add_argument("--output", default='data/coco/annotations/freemask_instances_pretrain2017.json')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--score-thr', type=float, default=0.3)
    parser.add_argument(
        "--split",
        type=int,
        default=0,
        help="Split id.",
    )
    args = parser.parse_args()
    return args

    

def main(args):
    mp.set_start_method("spawn", force=True)
    ann_dict = dict()
    ann_dict_ = json.load(open('data/coco/annotations/instances_train2017.json'))
    ann_dict['categories'] = ann_dict_['categories']

    images_list = []
    anns_list = []
    ann_id = 0

    imgs = glob.glob(args.input + '/*g')
    save_path = args.output
    
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, device=args.device)

    if isinstance(imgs, (list, tuple)):
        is_batch = True
    else:
        imgs = [imgs]
        is_batch = False

    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    if isinstance(imgs[0], np.ndarray):
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'

    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)

    datas = []
    for img in tqdm(imgs):
        img_path = img
        # prepare data
        if isinstance(img, np.ndarray):
            # directly add img
            data = dict(img=img)
        else:
            # add information into dict
            data = dict(img_info=dict(filename=img), img_prefix=None)
        # build the data pipeline
        data = test_pipeline(data)
        data = collate([data], samples_per_gpu=1)
        # just get the actual data from DataContainer
        data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
        data['img'] = [img.data[0] for img in data['img']]
        if next(model.parameters()).is_cuda:
            # scatter to specified GPU
            data = scatter(data, [device])[0]
        else:
            for m in model.modules():
                assert not isinstance(
                    m, RoIPool
                ), 'CPU inference with RoIPool is not supported currently.'

        # forward the model
        with torch.no_grad():
            keys = model.backbone(data['img'][0])[-1][0]
            C, height, width = keys.shape

            scale_factors = [1.0, 0.5, 0.25]
            queries_list = []
            for scale_factor in scale_factors:
                cur_queries = F.interpolate(keys[None, ...], scale_factor=scale_factor, mode='bilinear')[0].reshape(keys.shape[0], -1).permute(1, 0)
                num_q = len(cur_queries)
                queries_list.append(cur_queries)
            queries = torch.cat(queries_list)
            _, H, W = keys.shape
            keys = keys / keys.norm(dim=0, keepdim=True)
            queries = queries / queries.norm(dim=1, keepdim=True)
            attn = queries @ keys.reshape(keys.shape[0], -1)
            # normalize
            attn -= attn.min(-1, keepdim=True)[0]
            attn /= attn.max(-1, keepdim=True)[0]

            attn = attn.reshape(attn.shape[0], H, W)

            soft_masks = attn
            masks = soft_masks >= 0.5

            sum_masks = masks.sum((1,2))
            keep = sum_masks > 1
            if keep.sum() == 0:
                continue
            masks = masks[keep]
            soft_masks = soft_masks[keep]
            sum_masks = sum_masks[keep]
            queries = queries[keep]

            # Matrix NMS
            maskness = (soft_masks * masks.float()).sum((1, 2)) / sum_masks
            sort_inds = torch.argsort(maskness, descending=True)
            maskness = maskness[sort_inds]
            masks = masks[sort_inds]
            sum_masks = sum_masks[sort_inds]
            soft_masks = soft_masks[sort_inds]
            queries = queries[sort_inds]
            maskness = matrix_nms(masks, maskness*0, maskness, sigma=2, kernel='gaussian', sum_masks=sum_masks)


            sort_inds = torch.argsort(maskness, descending=True)
            if len(sort_inds) > 20:
                sort_inds = sort_inds[:20]
            masks = masks[sort_inds]
            maskness = maskness[sort_inds]
            soft_masks = soft_masks[sort_inds]
            queries = queries[sort_inds]

            soft_masks = F.interpolate(soft_masks[None, ...], size=(height, width), mode='bilinear')[0]
            masks = (soft_masks >= 0.5).float()
            sum_masks = masks.sum((1, 2))

            # mask to box
            width_proj = masks.max(1)[0]
            height_proj = masks.max(2)[0]
            box_width, box_height = width_proj.sum(1), height_proj.sum(1)
            center_ws, _ = center_of_mass(width_proj[:, None, :])
            _, center_hs = center_of_mass(height_proj[:, :, None])
            boxes = torch.stack([center_ws-0.5*box_width, center_hs-0.5*box_height, center_ws+0.5*box_width, center_hs+0.5*box_height], 1)
            #boxes = []
            #for mask in masks.cpu().numpy():
            #    ys, xs = np.where(mask)
            #    box = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
            #    boxes.append(box)
            #boxes = torch.tensor(boxes, device = maskness.device)

            # filter masks on the top border or with large width
            keep = center_hs > 0.2 * height
            keep_2 = (boxes[:, 2] - boxes[:, 0]) < 0.95 * width
            keep_3 = maskness >= 0.7
            keep = keep & keep_2 & keep_3
            #
            if keep.sum() == 0:
                continue
            masks = masks[keep]
            maskness = maskness[keep]
            boxes = boxes[keep]
            queries = queries[keep]

            # coco format
            img_name = img_path.split('/')[-1].split('.')[0]
            try:
                img_id = int(img_name)
            except:
                img_id = int(img_name.split('_')[-1])
            cur_image_dict = {'file_name': img_path.split('/')[-1],
                            'height': height,
                            'width': width,
                            'id':  img_id}
            images_list.append(cur_image_dict)


            masks = masks.cpu().numpy()
            maskness = maskness.cpu().numpy()
            boxes = boxes.tolist()
            queries = queries.tolist()
            rles = [mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0] for mask in masks]
            
            for idx in range(len(masks)):
                rle = rles[idx]
                rle['counts'] = rle['counts'].decode('ascii')
                cur_ann_dict = {'segmentation': rle,
                                'bbox': boxes[idx],
                                'score': float(maskness[idx]),
                                # 'emb': queries[idx],
                                'iscrowd': 0,
                                'image_id': img_id,
                                'category_id': 1,
                                'id':  ann_id}
                ann_id += 1
                anns_list.append(cur_ann_dict)


        ann_dict['images'] = images_list
        ann_dict['annotations'] = anns_list
        json.dump(ann_dict, open(save_path, 'w'))
        #json.dump(anns_list, open(save_path+'ann', 'w'))
    print("Done: {} images, {} annotations.".format(len(images_list), len(anns_list)))




if __name__ == '__main__':
    args = parse_args()
    main(args)

