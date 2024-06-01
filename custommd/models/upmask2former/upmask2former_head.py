'''
Author: lifuguan lifugan_10027@outlook.com
Date: 2022-07-14 16:34:51
LastEditors: lifuguan 1002732355@qq.com
LastEditTime: 2024-01-10 15:17:59
FilePath: /research_workspace/custommd/models/upmask2former/upmask2former_head.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import HEADS
from .mask2former_head import Mask2FormerHead

@HEADS.register_module()
class UpMask2FormerHead(Mask2FormerHead):
    def __init__(self,
                 in_channels,
                 feat_channels,
                 out_channels,
                 num_things_classes=80,
                 num_stuff_classes=53,
                 num_queries=100,
                 num_transformer_feat_level=3,
                 pixel_decoder=None,
                 enforce_decoder_input_project=False,
                 transformer_decoder=None,
                 positional_encoding=None,
                 loss_cls=None,
                 loss_mask=None,
                 loss_dice=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 num_proposals = 100,
                 query_shuffle=False,
                 is_kernel2query = True,
                 mask_ratio=0.05,
                 **kwargs):
        super(UpMask2FormerHead, self).__init__(in_channels, feat_channels, out_channels, num_things_classes, num_stuff_classes, num_queries, num_transformer_feat_level, pixel_decoder, enforce_decoder_input_project, transformer_decoder, positional_encoding, loss_cls, loss_mask, loss_dice, train_cfg, test_cfg, init_cfg)

        """build guided-kernel approach"""
        self.num_proposals = num_proposals
        self.is_kernel2query = is_kernel2query
        if self.is_kernel2query is True:
            self.query_shuffle = query_shuffle

            self.patch2query = nn.Linear(2048, 256)
            
            self.mask_ratio = mask_ratio
        
    def forward_train(self,
                      feats,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_masks,
                      gt_semantic_seg,
                      patch_feats,
                      gt_bboxes_ignore=None):
        # not consider ignoring bboxes
        assert gt_bboxes_ignore is None

        # forward
        all_cls_scores, all_mask_preds = self(feats, img_metas, patch_feats)

        # preprocess ground truth
        gt_labels, gt_masks = self.preprocess_gt(gt_labels, gt_masks,
                                                 gt_semantic_seg, img_metas)

        # loss
        losses = self.loss(all_cls_scores, all_mask_preds, gt_labels, gt_masks,
                           img_metas)

        return losses
    
    def forward(self, feats, img_metas, patch_feats):
        """Forward function.

        Args:
            feats (list[Tensor]): Multi scale Features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple: A tuple contains two elements.

            - cls_pred_list (list[Tensor)]: Classification logits \
                for each decoder layer. Each is a 3D-tensor with shape \
                (batch_size, num_queries, cls_out_channels). \
                Note `cls_out_channels` should includes background.
            - mask_pred_list (list[Tensor]): Mask logits for each \
                decoder layer. Each with shape (batch_size, num_queries, \
                 h, w).
        """
        batch_size = len(img_metas)
        mask_features, multi_scale_memorys = self.pixel_decoder(feats)
        # multi_scale_memorys (from low resolution to high resolution)
        decoder_inputs = []
        decoder_positional_encodings = []
        for i in range(self.num_transformer_feat_level):
            decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])
            # shape (batch_size, c, h, w) -> (h*w, batch_size, c)
            decoder_input = decoder_input.flatten(2).permute(2, 0, 1)
            level_embed = self.level_embed.weight[i].view(1, 1, -1)
            decoder_input = decoder_input + level_embed
            # shape (batch_size, c, h, w) -> (h*w, batch_size, c)
            mask = decoder_input.new_zeros(
                (batch_size, ) + multi_scale_memorys[i].shape[-2:],
                dtype=torch.bool)
            decoder_positional_encoding = self.decoder_positional_encoding(
                mask)
            decoder_positional_encoding = decoder_positional_encoding.flatten(
                2).permute(2, 0, 1)
            decoder_inputs.append(decoder_input)
            decoder_positional_encodings.append(decoder_positional_encoding)
        # shape (num_queries, c) -> (num_queries, batch_size, c)
        query_feat = self.query_feat.weight.unsqueeze(1).repeat(
            (1, batch_size, 1))
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(
            (1, batch_size, 1))

        bs = len(img_metas)

        if self.is_kernel2query is True:
            patch_feats = self.patch2query(patch_feats) \
                .view(bs, 20, -1) \
                .repeat_interleave(self.num_proposals // 20, dim=1) \
                .permute(1, 0, 2) \
                .contiguous()

            # with open('example.txt', 'w') as f:
            #     f.write(str(bs))
            #     f.write(str(patch_feats.shape))
            # c

            # if object query shuffle, we shuffle the index of object query embedding,
            # which simulate to adding patch feature to object query randomly.
            idx = torch.randperm(self.num_proposals) if self.query_shuffle else torch.arange(self.num_proposals)

            # NOTE query 
            # for training, it uses fixed number of query patches.
            mask_query_patch = (torch.rand(self.num_proposals, bs, 1, device=patch_feats.device) > self.mask_ratio).float()
            # NOTE mask some query patch and add query embedding
            refine_patch_feats = patch_feats * mask_query_patch
            query_feat = query_feat + refine_patch_feats[idx, ...]
                
        cls_pred_list = []
        mask_pred_list = []
        cls_pred, mask_pred, attn_mask = self.forward_head(
            query_feat, mask_features, multi_scale_memorys[0].shape[-2:])
        cls_pred_list.append(cls_pred)
        mask_pred_list.append(mask_pred)

        for i in range(self.num_transformer_decoder_layers):
            level_idx = i % self.num_transformer_feat_level
            # if a mask is all True(all background), then set it all False.
            attn_mask[torch.where(
                attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            # cross_attn + self_attn
            layer = self.transformer_decoder.layers[i]
            attn_masks = [attn_mask, None]
            query_feat = layer(
                query=query_feat,
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                query_pos=query_embed,
                key_pos=decoder_positional_encodings[level_idx],
                attn_masks=attn_masks,
                query_key_padding_mask=None,
                # here we do not apply masking on padded region
                key_padding_mask=None)
            cls_pred, mask_pred, attn_mask = self.forward_head(
                query_feat, mask_features, multi_scale_memorys[
                    (i + 1) % self.num_transformer_feat_level].shape[-2:])

            cls_pred_list.append(cls_pred)
            mask_pred_list.append(mask_pred)

        return cls_pred_list, mask_pred_list
