# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import math
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks

from .modeling.criterion import VideoSetCriterion
from .modeling.matcher import VideoHungarianMatcher
from .utils.memory import retry_if_cuda_oom
from scipy.optimize import linear_sum_assignment
import einops

logger = logging.getLogger(__name__)

# 打印指定 GPU 的使用内存
def print_gpu_memory_usage(gpu_id):
    # 选择 GPU
    torch.cuda.set_device(gpu_id)
    
    # 获取当前 GPU 的显存使用情况
    allocated_memory = torch.cuda.memory_allocated(gpu_id) / (1024 ** 2)  # 转换为 MB
    reserved_memory = torch.cuda.memory_reserved(gpu_id) / (1024 ** 2)    # 转换为 MB
    max_allocated = torch.cuda.max_memory_allocated(gpu_id) / (1024 ** 2)  # 历史最大分配量
    max_reserved = torch.cuda.max_memory_reserved(gpu_id) / (1024 ** 2)    # 历史最大预留量
    
    print(f"GPU {gpu_id}:")
    print(f"Allocated memory: {allocated_memory:.2f} MB")
    print(f"Reserved memory: {reserved_memory:.2f} MB")
    print(f"Max allocated memory: {max_allocated:.2f} MB")
    print(f"Max reserved memory: {max_reserved:.2f} MB")



@META_ARCH_REGISTRY.register()
class VideoMaskFormer(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # video
        num_frames,
        is_valid=False,
        classes=[],
        # prompt & continual & inference （from Mask2Former of ECLIPSE）
        softmask: bool = False,
        # inference
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        mask_bg: bool,
        test_topk_per_image: int,
        # continual
        continual: bool = False,
        # prompt tuning
        num_prompts: int = 0, 
        backbone_freeze: bool = False,
        cls_head_freeze: bool = False,
        mask_head_freeze: bool = False,
        pixel_decoder_freeze: bool = False,
        query_embed_freeze: bool = False,
        trans_decoder_freeze: bool = False,
        prompt_mask_mlp: bool = False,
        prompt_no_obj_mlp: bool = False,
        #lq
        cfg = None,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
        """
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        if criterion != None:
            self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        self.num_frames = num_frames

        self.is_valid = is_valid
        self.classes = classes
        # continual
        self.continual = continual
        self.model_old = False

        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.mask_bg = mask_bg
        self.test_topk_per_image = test_topk_per_image
        self.num_classes = sem_seg_head.num_classes
        
        # Parameters for ECLIPSE
        self.num_prompts = num_prompts
        self.backbone_freeze = backbone_freeze
        self.cls_head_freeze = cls_head_freeze
        self.mask_head_freeze = mask_head_freeze
        self.pixel_decoder_freeze = pixel_decoder_freeze
        self.query_embed_freeze = query_embed_freeze
        self.trans_decoder_freeze = trans_decoder_freeze
        self.prompt_mask_mlp = prompt_mask_mlp
        self.prompt_no_obj_mlp = prompt_no_obj_mlp

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference
        self.softmask = softmask
        self.cfg = cfg

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

        # building criterion
        matcher = VideoHungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]

        criterion = VideoSetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        )

        continual = hasattr(cfg, "CONT") # （from Mask2Former of ECLIPSE）
        meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        if not continual:
            if not cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
                # Loss parameters:
                criterion = criterion
            else:
                criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=meta.ignore_label)
        else:
            criterion = criterion # 修改criterion，改为None则使用VideoMaskFormerDistillation

        ret =  {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": True,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # video
            "num_frames": cfg.INPUT.SAMPLING_FRAME_NUM,
            # from （from Mask2Former of ECLIPSE）
            # "per_pixel": cfg.MODEL.MASK_FORMER.PER_PIXEL and cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
            "softmask": cfg.MODEL.MASK_FORMER.SOFTMASK,
            # inference
            "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
            "mask_bg": cfg.MODEL.MASK_FORMER.TEST.MASK_BG and cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            # continual
            "continual": continual,
            # prompt tuning
            "num_prompts": cfg.CONT.NUM_PROMPTS,
            "backbone_freeze": cfg.CONT.BACKBONE_FREEZE,
            "cls_head_freeze": cfg.CONT.CLS_HEAD_FREEZE,
            "mask_head_freeze": cfg.CONT.MASK_HEAD_FREEZE,
            "pixel_decoder_freeze": cfg.CONT.PIXEL_DECODER_FREEZE,
            "query_embed_freeze": cfg.CONT.QUERY_EMBED_FREEZE,
            "trans_decoder_freeze": cfg.CONT.TRANS_DECODER_FREEZE,
            "prompt_mask_mlp": cfg.CONT.PROMPT_MASK_MLP,
            "prompt_no_obj_mlp": cfg.CONT.PROMPT_NO_OBJ_MLP,
            "cfg": cfg,
        }

        return ret

    @property
    def device(self):
        return self.pixel_mean.device
    
    def forward_inference(self, images, batched_inputs, outputs):
        mask_cls_results = outputs["pred_logits"]
        mask_pred_results = outputs["pred_masks"] if self.softmask else outputs["pred_masks"]
        # upsample masks
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(images.tensor.shape[-2], images.tensor.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )

        del outputs

        processed_results = []
        for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
        ):
            if 'sem_seg' in input_per_image and self.continual and self.semantic_on:
                height, width = input_per_image['sem_seg'].shape
            else:
                height = input_per_image.get("height", image_size[0])  # image_size[0]
                width = input_per_image.get("width", image_size[1])  # image_size[1]
            processed_results.append({})

            if self.sem_seg_postprocess_before_inference:
                # That's interpolation to image size
                mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                    mask_pred_result, image_size, height, width
                )
                mask_cls_result = mask_cls_result.to(mask_pred_result)

            # semantic segmentation inference
            if self.semantic_on:
                r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                if not self.sem_seg_postprocess_before_inference:
                    r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                processed_results[-1]["sem_seg"] = r

            # panoptic segmentation inference
            if self.panoptic_on:
                panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
                processed_results[-1]["panoptic_seg"] = panoptic_r

            # instance segmentation inference
            if self.instance_on:
                instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result)
                processed_results[-1]["instances"] = instance_r

        return processed_results

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        # if input is a list of Datamapper
        if isinstance(batched_inputs,list):# list of dict
            images = []
            for video in batched_inputs:
                for frame in video["image"]:
                    images.append(frame.to(self.device))
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
            images = ImageList.from_tensors(images, self.size_divisibility)
            targets = None
        elif isinstance(batched_inputs, tuple):# (image, target)
 
            if isinstance(batched_inputs[0], torch.Tensor):
                images = batched_inputs[0]
            else:
                print('images is not Tensor!')
            targets = batched_inputs[1]
        elif isinstance(batched_inputs, torch.Tensor):
            if isinstance(batched_inputs, torch.Tensor):
                images = [batched_inputs.to(self.device)]
                images = [(x - self.pixel_mean) / self.pixel_std for x in images]
                images = ImageList.from_tensors(images, self.size_divisibility)
            else:
                print('images is not Tensor!')
            targets = None
        
        images_tensor = images.tensor if isinstance(images, ImageList) else images
        if images_tensor.ndim != 4:
            images_tensor = images_tensor.squeeze(0)
        
        features = self.backbone(images_tensor)
        # if self.is_valid == True:
        #     self.sem_seg_head.predictor.is_valid = True
        # else:
        #     self.sem_seg_head.predictor.is_valid = False
        outputs = self.sem_seg_head(features)
        
        '''train or valid '''
        if self.training or self.is_valid:
            # mask classification target
            if targets == None:
                targets = self.prepare_targets(batched_inputs, images)
                
            if not self.model_old and self.criterion!=None:
                criterion_out = self.criterion(outputs, targets)
                losses, indices = criterion_out["losses"], criterion_out["indices"]
                if outputs['loss_contrastive'] is not None and self.cfg.CRISP.USE_CLIP_PROMPTER:
                    contrastive_loss = outputs['loss_contrastive']
                    losses.update({'loss_contrastive': contrastive_loss})
                    if self.cfg.CRISP.USE_CONTRASTIVE_LOSS:
                        self.criterion.weight_dict.update({'loss_contrastive': self.cfg.CRISP.CONTRASTIVE_WEIGHT})
                if outputs['loss_orth'] is not None:
                    orth_loss = outputs['loss_orth']
                    losses.update(orth_loss)
                    if self.cfg.CRISP.USE_ORTH_LOSS:
                        for k in orth_loss.keys():
                            self.criterion.weight_dict.update({k: self.cfg.CRISP.ORTH_WEIGHT})
                del indices
                
                for k in list(losses.keys()):
                    if k in self.criterion.weight_dict:
                        losses[k] *= self.criterion.weight_dict[k]
                    else:
                        # remove this loss if not specified in `weight_dict`
                        losses.pop(k)
                return {"losses":losses,
                        "outputs": outputs,
                        "features":features,
                        "targets": targets}
            else: 
                return outputs
            
        else:
            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]
            decoder_feat = outputs["decoder_feat"]

            mask_cls_result = mask_cls_results[0]
            # upsample masks
            assert isinstance(images, ImageList), "ERROR from VideoMaskFormer, images is not ImageList"
            
            mask_pred_result = retry_if_cuda_oom(F.interpolate)(
                mask_pred_results[0],
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )
            input_per_image = batched_inputs[0]
            image_size = images.image_sizes[0]  # image size without padding after data augmentation

            height = input_per_image.get("height", image_size[0])  # raw image size before data augmentation
            width = input_per_image.get("width", image_size[1])
            video_output = retry_if_cuda_oom(self.inference_video)(mask_cls_result, mask_pred_result, image_size, height, width)
            assert video_output != None, "Error video_output is None"

            if targets == None:
                targets = self.prepare_targets_valid(batched_inputs, images)
            video_output.update({
                "decoder_feat": decoder_feat,
                "targets": targets,
                "att_map": outputs['att_map'] # attention map
            })
            return video_output

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        gt_instances = []
        for targets_per_video in targets:
            _num_instance = len(targets_per_video["instances"][0])
            mask_shape = [_num_instance, self.num_frames, h_pad, w_pad]
            gt_masks_per_video = torch.zeros(mask_shape, dtype=torch.bool, device=self.device)

            gt_ids_per_video = []
            for f_i, targets_per_frame in enumerate(targets_per_video["instances"]):
                targets_per_frame = targets_per_frame.to(self.device)
                h, w = targets_per_frame.image_size

                gt_ids_per_video.append(targets_per_frame.gt_ids[:, None])
                gt_masks_per_video[:, f_i, :h, :w] = targets_per_frame.gt_masks.tensor

            gt_ids_per_video = torch.cat(gt_ids_per_video, dim=1)
            valid_idx = (gt_ids_per_video != -1).any(dim=-1)

            gt_classes_per_video = targets_per_frame.gt_classes[valid_idx]          # N,
            gt_ids_per_video = gt_ids_per_video[valid_idx]                          # N, num_frames

            gt_instances.append({"labels": gt_classes_per_video, "ids": gt_ids_per_video})
            gt_masks_per_video = gt_masks_per_video[valid_idx].float()          # N, num_frames, H, W
            gt_instances[-1].update({"masks": gt_masks_per_video})

        return gt_instances
    def prepare_targets_valid(self, targets, images):
        h_pad, w_pad = (targets[0]['height'], targets[0]['width'])
        gt_instances = []
        for targets_per_video in targets:
            _num_instance = len(targets_per_video["instances"][0])
            mask_shape = [_num_instance, targets_per_video['length'], h_pad, w_pad]
            gt_masks_per_video = torch.zeros(mask_shape, dtype=torch.bool, device=self.device)

            gt_ids_per_video = []
            for f_i, targets_per_frame in enumerate(targets_per_video["instances"]):
                targets_per_frame = targets_per_frame.to(self.device)
                h, w = h_pad, w_pad

                gt_ids_per_video.append(targets_per_frame.gt_ids[:, None])
                temp = targets_per_frame.gt_masks.tensor.unsqueeze(0).float()
                gt = F.interpolate(
                    temp, size=(h_pad, w_pad), mode="bilinear", align_corners=False
                    ).squeeze().round().int()
                gt_masks_per_video[:, f_i, :h, :w] = gt

            gt_ids_per_video = torch.cat(gt_ids_per_video, dim=1)
            valid_idx = (gt_ids_per_video != -1).any(dim=-1)

            gt_classes_per_video = targets_per_frame.gt_classes[valid_idx]          # N,
            gt_ids_per_video = gt_ids_per_video[valid_idx]                          # N, num_frames

            gt_instances.append({"labels": gt_classes_per_video, "ids": gt_ids_per_video})
            gt_masks_per_video = gt_masks_per_video[valid_idx].float()          # N, num_frames, H, W
            gt_instances[-1].update({"masks": gt_masks_per_video})

        return gt_instances

    def inference_video(self, pred_cls, pred_masks, img_size, output_height, output_width):
        if len(pred_cls) > 0:

            scores = F.softmax(pred_cls, dim=-1)[:, :-1]#100*num_class
            num_classes = scores.shape[-1]
            # labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)# 0~num_class-1
            labels = torch.arange(num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)# 0~num_class-1
            num_queries = scores.shape[0]
            labels = torch.arange(num_classes, device=self.device).unsqueeze(0).repeat(num_queries, 1).flatten(0, 1)# 0~num_class-1
            # keep top-10 predictions
            scores_per_image, topk_indices = scores.flatten(0, 1).topk(10, sorted=False)
            labels_per_image = labels[topk_indices]
            # topk_indices = topk_indices // self.sem_seg_head.num_classes
            topk_indices = topk_indices // num_classes
            pred_masks = pred_masks.to('cuda')
            pred_masks = pred_masks[topk_indices]
            # print(pred_masks.shape, )
            if not isinstance(img_size[0], int):
                img_size = img_size[0]
            pred_masks = pred_masks[:, :, : img_size[0], : img_size[1]]
            pred_masks = F.interpolate(
                pred_masks, size=(output_height, output_width), mode="bilinear", align_corners=False
            )

            masks = pred_masks > 0.
            out_scores = scores_per_image.cpu().tolist()
            out_labels = labels_per_image.cpu().tolist()
            out_masks = [m for m in masks.cpu()]
        else:
            out_scores = []
            out_labels = []
            out_masks = []

        video_output = {
            "image_size": (output_height, output_width),
            "pred_logits": pred_cls,
            "pred_scores": out_scores,
            "pred_labels": out_labels,
            "pred_masks": out_masks,
            "topk_indices": topk_indices,
        }

        return video_output
    
    def init_new_classifier(self,old_class_embed_paras, device):
        
        cls = self.cls
        imprinting_w = old_class_embed_paras
        bkg_bias = old_class_embed_paras.bias[0]

        bias_diff = torch.log(torch.FloatTensor([self.classes[-1] + 1])).to(device)

        new_bias = (bkg_bias - bias_diff)

        cls.weight.data.copy_(imprinting_w)
        cls.bias.data.copy_(new_bias)

        self.cls[0].bias[0].data.copy_(new_bias.squeeze(0))
    
    def make_pseudolabels2(self, out, data, targets):
        img_size = data[0]['image'].shape[-2], data[0]['image'].shape[-1]
        logits, mask = out['outputs']['pred_logits'], out['outputs']['pred_masks']  # tensors of size BxQxK, BxQxHxW
        mask = F.interpolate(
            mask,
            size=img_size,
            mode="bilinear",
            align_corners=False,
        )

        for i in range(logits.shape[0]):  # iterate on batch size
            scores, labels = F.softmax(logits[i], dim=-1).max(-1)
            mask_pred = mask[i].sigmoid() if not self.softmask else mask[i].softmax(dim=0)

            keep = labels.ne(self.old_classes) & (scores > self.pseudo_thr)
            cur_scores = scores[keep]
            cur_classes = labels[keep]
            cur_masks = mask_pred[keep]
            cur_masks_bin = mask_pred[keep].clone()

            cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

            tar = targets[i]
            gt_pixels = tar['masks'].sum(dim=0).bool()  # H,W
            keep2 = torch.zeros(len(cur_masks)).bool()

            if cur_masks.shape[0] > 0:
                # take argmax
                cur_mask_ids = cur_prob_masks.argmax(0)  # REMOVE GT
                cur_mask_ids[gt_pixels] = -1

                for k in range(cur_classes.shape[0]):
                    mask_area = (cur_mask_ids == k).sum().item()
                    original_area = (cur_masks[k] >= 0.5).sum().item()
                    x_mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                    if mask_area > 0 and original_area > 0 and x_mask.sum().item() > 0:
                        if mask_area / original_area > 0.5:
                            keep2[k] = 1
                            cur_masks_bin[k] = x_mask

            if keep2.sum() > 0:
                pseudo_lab = cur_classes[keep2]
                pseudo_mask = cur_masks_bin[keep2].bool()

                tar['masks'] = torch.cat((tar['masks'], pseudo_mask), dim=0)
                tar['labels'] = torch.cat((tar['labels'], pseudo_lab), dim=0)

        return targets
    
    def freeze_for_prompt_tuning(self, ):
        """
        ECLIPSE: freeze parameters for visual prompt tuning
        """
        if not self.model_old:
            
            for k, p in self.named_parameters():
                p.requires_grad = False
                
            # unfreeze last classifier layer
            for k, p in self.sem_seg_head.predictor.class_embed.cls[-1].named_parameters():
                p.requires_grad = True
                
            # unfreeze cls-layer for no-obj
            for k, p in self.sem_seg_head.predictor.class_embed.cls[0].named_parameters():
                p.requires_grad = True
                
            if self.num_prompts > 0:
                # unfreeze prompt embeddings
                for k, p in self.sem_seg_head.predictor.prompt_feat[-1].named_parameters():
                    p.requires_grad = True
                for k, p in self.sem_seg_head.predictor.prompt_embed[-1].named_parameters():
                    p.requires_grad = True
                
            if not self.backbone_freeze:
                for k, p in self.backbone.named_parameters():
                    p.requires_grad = True
                    
            if not self.trans_decoder_freeze and self.num_prompts > 0:
                for k, p in self.sem_seg_head.predictor.prompt_transformer_self_attention_layers[-1].named_parameters():
                    p.requires_grad = True
                for k, p in self.sem_seg_head.predictor.prompt_transformer_cross_attention_layers[-1].named_parameters():
                    p.requires_grad = True
                for k, p in self.sem_seg_head.predictor.prompt_transformer_ffn_layers[-1].named_parameters():
                    p.requires_grad = True
                    
            if not self.pixel_decoder_freeze:
                for k, p in self.sem_seg_head.pixel_decoder.named_parameters():
                    p.requires_grad = True

            if not self.cls_head_freeze:
                for k, p in self.sem_seg_head.predictor.class_embed.cls.named_parameters():
                    p.requires_grad = True

            if not self.mask_head_freeze:
                for k, p in self.sem_seg_head.predictor.mask_embed.named_parameters():
                    p.requires_grad = True
                    
            if not self.query_embed_freeze:
                for k, p in self.sem_seg_head.predictor.query_feat.named_parameters():
                    p.requires_grad = True
                for k, p in self.sem_seg_head.predictor.query_embed.named_parameters():
                    p.requires_grad = True
                    
            if self.prompt_mask_mlp and self.num_prompts > 0:
                for k, p in self.sem_seg_head.predictor.prompt_mask_embed[-1].named_parameters():
                    p.requires_grad = True
                    
                for k, p in self.sem_seg_head.predictor.mask_embed.named_parameters():
                    p.requires_grad = False
                    
            if self.prompt_no_obj_mlp and self.num_prompts > 0:
                for k, p in self.sem_seg_head.predictor.prompt_no_obj_embed[-1].named_parameters():
                    p.requires_grad = True
                    
                for k, p in self.sem_seg_head.predictor.class_embed.cls[0].named_parameters():
                    p.requires_grad = False
                    
            if self.num_prompts == 0 and not self.use_appearance_decoder:
                # unfreeze prompt embeddings
                for k, p in self.sem_seg_head.predictor.query_feat.named_parameters():
                    p.requires_grad = True
                for k, p in self.sem_seg_head.predictor.query_embed.named_parameters():
                    p.requires_grad = True
                for k, p in self.sem_seg_head.predictor.mask_embed.named_parameters():
                    p.requires_grad = True
            if self.num_prompts > 0 and hasattr(self.sem_seg_head.predictor, "sem_prompter"):
                self.sem_seg_head.predictor.sem_prompter.learnable_prompts[-1].weight.requires_grad = True
                
    def copy_prompt_embed_weights(self, ):
        if self.num_prompts > 0:
            base_feat_weights = self.sem_seg_head.predictor.query_feat.weight
            base_embed_weights = self.sem_seg_head.predictor.query_embed.weight
            
            base_feat_weights = base_feat_weights.mean(0, keepdims=True).repeat(self.num_prompts, 1)
            base_embed_weights = base_embed_weights.mean(0, keepdims=True).repeat(self.num_prompts, 1)
            
            self.sem_seg_head.predictor.prompt_feat[-1].load_state_dict({"weight": base_feat_weights})
            
            if isinstance(self.sem_seg_head.predictor.prompt_embed[-1], nn.ModuleList):
                embed_dict = {}
                for n in range(len(self.sem_seg_head.predictor.prompt_embed[-1])):
                    embed_dict[f"{n}.weight"] = base_embed_weights
            else:
                embed_dict = {"weight": base_embed_weights}
                
            self.sem_seg_head.predictor.prompt_embed[-1].load_state_dict(embed_dict)
                
            
    def copy_mask_embed_weights(self, ):
        if self.num_prompts > 0 and self.prompt_mask_mlp:
            self.sem_seg_head.predictor.prompt_mask_embed[-1].load_state_dict(
                self.sem_seg_head.predictor.mask_embed.state_dict()
            )
            
            
    def copy_prompt_trans_decoder_weights(self, ):
        if self.num_prompts > 0 and not self.trans_decoder_freeze:
            self.sem_seg_head.predictor.prompt_transformer_self_attention_layers[-1].load_state_dict(
                self.sem_seg_head.predictor.transformer_self_attention_layers.state_dict()
            )
            self.sem_seg_head.predictor.prompt_transformer_cross_attention_layers[-1].load_state_dict(
                self.sem_seg_head.predictor.transformer_cross_attention_layers.state_dict()
            )
            self.sem_seg_head.predictor.prompt_transformer_ffn_layers[-1].load_state_dict(
                self.sem_seg_head.predictor.transformer_ffn_layers.state_dict()
            )
            
            
    def copy_no_obj_weights(self, ):
        if self.num_prompts > 0:
            no_obj_weights = self.sem_seg_head.predictor.class_embed.cls[0].state_dict()
            novel_cls_weights = self.sem_seg_head.predictor.class_embed.cls[-1].state_dict()

            for k, v in no_obj_weights.items():
                if v.shape == novel_cls_weights[k].shape:
                    novel_cls_weights[k] = v
                else:
                    if "weight" in k:
                        novel_cls_weights[k] = v.repeat(novel_cls_weights[k].shape[0], 1)
                    elif "bias" in k:
                        novel_cls_weights[k] = v.repeat(novel_cls_weights[k].shape[0])
            self.sem_seg_head.predictor.class_embed.cls[-1].load_state_dict(novel_cls_weights)


    def match_from_embds(self, prevs, curs, group_match=False):
        cur_embds, cur_app_embds = curs
        tgt_embds, tgt_app_embds = prevs

        cur_embds = cur_embds / cur_embds.norm(dim=1)[:, None]
        tgt_embds = tgt_embds / tgt_embds.norm(dim=1)[:, None]
        cos_sim = torch.mm(cur_embds, tgt_embds.transpose(0,1))

        cur_app_embds = cur_app_embds / cur_app_embds.norm(dim=1)[:, None]
        tgt_app_embds = tgt_app_embds / tgt_app_embds.norm(dim=1)[:, None]
        cos_sim_app = torch.mm(cur_app_embds, tgt_app_embds.transpose(0,1))

        cost_embd = (1 - self.appearance_weight) * cos_sim + self.appearance_weight * cos_sim_app

        C = 1.0 * (cost_embd)
        C = C.cpu()
        if not group_match:
            indices = linear_sum_assignment(C.transpose(0, 1), maximize=True)  # target x current
            indices = indices[1]  # permutation that makes current aligns to target
        return indices
    
    def match_from_embds_with_likelihood(self, prevs, curs, group_match=False):
        cur_embds, cur_app_embds = curs
        tgt_embds, tgt_app_embds = prevs

        cur_embds = cur_embds / cur_embds.norm(dim=1)[:, None]
        tgt_embds = tgt_embds / tgt_embds.norm(dim=1)[:, None]
        cos_sim = torch.mm(cur_embds, tgt_embds.transpose(0,1))

        cur_app_embds = cur_app_embds / cur_app_embds.norm(dim=1)[:, None]
        tgt_app_embds = tgt_app_embds / tgt_app_embds.norm(dim=1)[:, None]
        cos_sim_app = torch.mm(cur_app_embds, tgt_app_embds.transpose(0,1))

        cost_embd = (1 - self.appearance_weight) * cos_sim + self.appearance_weight * cos_sim_app

        C = 1.0 * (cost_embd)
        C = C.cpu()
        if not group_match:
            indices = linear_sum_assignment(C.transpose(0, 1), maximize=True)  # target x current
            indices = indices[1]  # permutation that makes current aligns to target
        else:
            indices_list = []
            for i in range(3):
                remove_ids = self.merge_sublists_by_concatenation(indices_list) if i>0 else None
                cur_indices = linear_sum_assignment(C.transpose(0, 1), maximize=True)
                if remove_ids is not None:
                    for idx in remove_ids:
                        C[idx] = 1e5
                indices_list.append(cur_indices)
            return indices_list
        return indices

    def post_processing(self, outputs, reid_queries, appearance_queries):
        pred_logits, pred_masks, pred_embds = outputs['pred_logits'], outputs['pred_masks'], outputs['pred_embds']
        self.memory_bank.reset()
        self.appearance_memory_bank.reset()

        # pred_logits: 1 t q c
        # pred_masks: 1 q t h w
        pred_logits = pred_logits[0]
        pred_masks = einops.rearrange(pred_masks[0], 'q t h w -> t q h w')
        pred_embds = pred_embds[0]
        reid_queries = reid_queries[0]
        appearance_queries = appearance_queries[0]

        pred_logits = list(torch.unbind(pred_logits))
        pred_masks = list(torch.unbind(pred_masks))
        pred_embds = list(torch.unbind(pred_embds))
        reid_queries = list(torch.unbind(reid_queries))
        appearance_queries = list(torch.unbind(appearance_queries))

        out_logits = []
        out_masks = []
        out_logits.append(pred_logits[0])
        out_masks.append(pred_masks[0])
        max_scores, _ = torch.max(out_logits[-1].softmax(dim=-1)[:, :-1], dim=-1)
        self.memory_bank.update(reid_queries[0], max_scores)
        self.appearance_memory_bank.update(appearance_queries[0], max_scores)

        for i in range(1, len(pred_logits)):
            prevs = (self.memory_bank.get(), self.appearance_memory_bank.get())
            curs = (reid_queries[i], appearance_queries[i])
            indices = self.match_from_embds(prevs, curs)
            # 
            out_logits.append(pred_logits[i][indices, :])
            out_masks.append(pred_masks[i][indices, :, :])
            max_scores, _ = torch.max(out_logits[-1].softmax(dim=-1)[:, :-1], dim=-1)
            self.memory_bank.update(reid_queries[i][indices, :], max_scores)
            self.appearance_memory_bank.update(appearance_queries[i][indices, :], max_scores)

        out_logits = sum(out_logits)/len(out_logits)
        out_masks = torch.stack(out_masks, dim=1)  # q h w -> q t h w

        out_logits = out_logits.unsqueeze(0)
        out_masks = out_masks.unsqueeze(0)

        outputs['pred_logits'] = out_logits
        outputs['pred_masks'] = out_masks

        return outputs

    def run_window_inference(self, images_tensor, window_size=20):
        iters = len(images_tensor) // window_size
        if len(images_tensor) % window_size != 0:
            iters += 1
        out_list = []
        reid_list = []
        appearance_list = []
        for i in range(iters):
            start_idx = i * window_size
            end_idx = (i+1) * window_size

            features = self.backbone(images_tensor[start_idx:end_idx])
            out = self.sem_seg_head(features)
            appearance_features = [features[f].detach() for f in self.appearance_in_features]
            reid_queries, appearance_queries = self.appearance_decoder(out['pred_embds'], appearance_features, einops.rearrange(out['pred_masks'], 'b q t h w -> (b t) q () h w'))
            del features['res2'], features['res3'], features['res4'], features['res5'], appearance_features
            for j in range(len(out['aux_outputs'])):
                del out['aux_outputs'][j]['pred_masks'], out['aux_outputs'][j]['pred_logits']
            out_list.append(out)
            reid_list.append(reid_queries)
            appearance_list.append(appearance_queries)

        # merge outputs
        outputs = {}
        outputs['pred_logits'] = torch.cat([x['pred_logits'] for x in out_list], dim=1).detach()
        outputs['pred_masks'] = torch.cat([x['pred_masks'] for x in out_list], dim=2).detach()
        outputs['pred_embds'] = torch.cat([x['pred_embds'] for x in out_list], dim=1).detach()

        return outputs, reid_queries, appearance_queries