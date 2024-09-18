# Saliency Prompt
This code contains our extended version based on CVPR.
## Timeline
:triangular_flag_on_post: **Updates**
- Releasing `Unsupervised Pre-training with Language-Vision Prompts for Low-Data Instance Segmentation`.
- Our `Boosting low-data instance segmentation by unsupervised pre-training with saliency prompt` has been accepted by CVPR 2023.

```bibtex
@article{zhang2024unsupervised,
  title={Unsupervised Pre-training with Language-Vision Prompts for Low-Data Instance Segmentation},
  author={Zhang, Dingwen and Li, Hao and He, Diqi and Liu, Nian and Cheng, Lechao and Wang, Jingdong and Han, Junwei},
  journal={arXiv preprint arXiv:2405.13388},
  year={2024}
}
@inproceedings{li2023boosting,
  title={Boosting low-data instance segmentation by unsupervised pre-training with saliency prompt},
  author={Li, Hao and Zhang, Dingwen and Liu, Nian and Cheng, Lechao and Dai, Yalun and Zhang, Chao and Wang, Xinggang and Han, Junwei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={15485--15494},
  year={2023}
}
```

## Requirements

* Python 3.8
* CUDA 11.3
* PyTorch 1.10.0
* mmdet 2.19.0
* mmcv-full 1.4.8
* mmselfsup 0.10.0


## File Structure

    UPLVP/
    ├── configs/
    ├── custommd/
    ├── tools_det/
    README.md

## Usage
### Mask Proposal
Requirements:
* clip 1.0
* tensorflow-gpu 2.6.0
* pycocotools 2.0.0

To generate pseudo masks run:
```
python tools_det/maskproposal.py
```

### Pre-train
To pre-train K-Net/Mask2former/QueryInst with 8 gpus run:
```bash
bash tools_det/dist_train.sh configs/selfsup/upknet/upknet_feature_coco_pretrain_labeled_prompt_ann.py 8

or

bash tools_det/dist_train.sh configs/selfsup/upmask2former/upmask2former_r50_lsj_8x2_50e_coco_prompt_ann.py 8

or

bash tools_det/dist_train.sh configs/selfsup/upqueryinst/upqueryinst_r50_fpn_1x_coco_pretrain_moco_labeled_prompt.py 8
```

### Fine-tune
To fine-tune K-Net with 8 gpus on COCO-10%/Cityscapes/CTW1500 run:
```bash
bash tools_det/dist_train.sh configs/det/knet/upknet_inherit_coco_10image_openseg_ann.py 8

or

bash tools_det/dist_train.sh configs/det/knet/upknet_s3_r50_fpn_1x_cityscapes.py 8

or

bash tools_det/dist_train.sh configs/det/knet/upknet_s3_r50_fpn_1x_ctw1500.py 8
```

## Acknowledgement

This work was supported in part by the National Natural Science Foundation of China under Grant 62293543, Grant U21B2048 and Grant 62106235.


    