from .base import BaseDistillation
from mask2former.maskformer_model import MaskFormer
from mask2former_video.video_maskformer_model import VideoMaskFormer
from mask2former.modeling.matcher import HungarianMatcher, SoftmaxMatcher
from mask2former_video.modeling.matcher import VideoHungarianMatcher
from .set_criterion import KDSetCriterion, SoftmaxKDSetCriterion
from .set_pseudo import PseudoSetCriterion
import torch
import torch.nn.functional as F
from continual.modeling.pod import func_pod_loss
from mask2former_video.modeling.criterion import VideoSetCriterion, VideoSetCriterion_incremental
from detectron2.structures import Boxes, ImageList

def pod_loss(output, output_old):
    # fixme actually pod is computed BEFORE ReLU. Due to Detectron, it is hard to move after...
    input_feat = output['features']
    input_feat = [input_feat[key] for key in ['res2', 'res3', 'res4', 'res5']] + [output['outputs']['features']]

    old_feat = output_old['features']
    old_feat = [old_feat[key].detach() for key in ['res2', 'res3', 'res4', 'res5']] + [output_old['outputs']['features']]

    loss = {"loss_pod": func_pod_loss(input_feat, old_feat, scales=[1, 2, 4])}
    return loss


class VideoMaskFormerDistillation(BaseDistillation):
    def __init__(self, cfg, model, model_old):
        super().__init__(cfg, model, model_old)

        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        cx = cfg.MODEL.MASK_FORMER
        class_weight = cx.CLASS_WEIGHT
        dice_weight = cx.DICE_WEIGHT
        mask_weight = cx.MASK_WEIGHT
        self.no_object_weight = no_object_weight

        self.use_kd = cfg.CONT.TASK and cfg.CONT.DIST.KD_WEIGHT > 0
        self.kd_weight = cfg.CONT.DIST.KD_WEIGHT if self.use_kd else 0.
        self.pod_weight = cfg.CONT.DIST.POD_WEIGHT

        self.pseudolabeling = cfg.CONT.DIST.PSEUDO
        self.pseudo_type = cfg.CONT.DIST.PSEUDO_TYPE
        self.pseudo_thr = cfg.CONT.DIST.PSEUDO_THRESHOLD
        self.alpha = cfg.CONT.DIST.ALPHA
        self.pseudo_mask_threshold = 0.5
        self.iou_threshold = cfg.CONT.DIST.IOU_THRESHOLD

        self.softmask = cfg.MODEL.MASK_FORMER.SOFTMASK
        if not isinstance(model, torch.nn.parallel.DistributedDataParallel):
            self.num_classes = sum(model.classes) + 1
            self.new_classes = model.classes[-1]
            self.old_classes = sum(model.classes[:-1])
        else:
            self.num_classes = sum(model.module.classes) + 1
            self.new_classes = model.module.classes[-1]
            self.old_classes = sum(model.module.classes[:-1])
        # building criterion
        if self.softmask:
            matcher = SoftmaxMatcher(
                cost_class=class_weight,
                cost_dice=dice_weight,
                num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            )
        else:
            matcher = HungarianMatcher(
                cost_class=class_weight,
                cost_mask=mask_weight,
                cost_dice=dice_weight,
                num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            )
            # matcher = VideoHungarianMatcher(
            #     cost_class=class_weight,
            #     cost_mask=mask_weight,
            #     cost_dice=dice_weight,
            #     num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            # )
        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight,
                       "loss_dice": dice_weight, "loss_kd": self.kd_weight, "loss_mask_kd": cfg.CONT.DIST.MASK_KD,
                       "loss_pod": cfg.CONT.DIST.POD_WEIGHT * (self.new_classes / self.num_classes)**0.5}

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]

        # criterion = SoftmaxKDSetCriterion if self.softmask else KDSetCriterion
        Criterion = SoftmaxKDSetCriterion if self.softmask else VideoSetCriterion_incremental
        if isinstance(Criterion, SoftmaxKDSetCriterion) or isinstance(Criterion, KDSetCriterion):
            self.criterion = Criterion(
                self.num_classes,
                matcher=matcher,
                # Parameters for learning new classes
                weight_dict=weight_dict,
                eos_coef=no_object_weight,
                focal=cfg.MODEL.MASK_FORMER.FOCAL,
                focal_alpha=cfg.MODEL.MASK_FORMER.FOCAL_ALPHA, focal_gamma=cfg.MODEL.MASK_FORMER.FOCAL_GAMMA,
                losses=losses,
                num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
                oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
                importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
                deep_sup=cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION,
                # Parameters for not forget
                old_classes=self.old_classes, use_kd=self.use_kd, use_bkg=self.use_bg,
                uce=cfg.CONT.DIST.UCE, ukd=cfg.CONT.DIST.UKD, l2=cfg.CONT.DIST.L2, kd_deep=cfg.CONT.DIST.KD_DEEP,
                alpha=cfg.CONT.DIST.ALPHA, kd_reweight=cfg.CONT.DIST.KD_REW, mask_kd=cfg.CONT.DIST.MASK_KD,
            )
        else:
            if not isinstance(model, torch.nn.parallel.DistributedDataParallel):
                self.criterion = Criterion(
                        model.sem_seg_head.num_classes,
                        matcher=matcher,
                        weight_dict=weight_dict,
                        eos_coef=no_object_weight,
                        losses=losses,
                        num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
                        oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
                        importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
                    )
            else:
                self.criterion = Criterion(
                        model.module.sem_seg_head.num_classes,
                        matcher=matcher,
                        weight_dict=weight_dict,
                        eos_coef=no_object_weight,
                        losses=losses,
                        num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
                        oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
                        importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
                    )
        self.criterion.to(self.device)

    def make_pseudolabels2(self, out, data, targets):
        # img_size = data[0]['image'][0].shape[-2], data[0]['image'][0].shape[-1]
        size = targets[0]['masks'].shape[-2], targets[0]['masks'].shape[-1]
        logits, mask = out['pred_logits'], out['pred_masks']  # tensors of size BxQxK, BxQx2xHxW
        mask_list = []
        for i in range(mask.shape[0]):
            _mask = F.interpolate(
                mask[i],
                size=size,
                mode="bilinear",
                align_corners=False,
            )
            mask_list.append(_mask)
        mask_shape = out['pred_masks'].shape
        _B, _Q, num_frames, _H, _W = mask_shape[0], mask_shape[1], mask_shape[2] ,size[0], size[1]
        mask = torch.cat(mask_list).reshape((_B, _Q, num_frames, _H, _W))
        
        for i in range(logits.shape[0]):  # iterate on batch size
            scores, labels = F.softmax(logits[i], dim=-1).max(-1)
            mask_pred = mask[i].sigmoid() if not self.softmask else mask[i].softmax(dim=0)

            keep = labels.ne(self.old_classes) & (scores > self.pseudo_thr) # don't use bg
            cur_scores = scores[keep]
            cur_classes = labels[keep]
            cur_masks = mask_pred[keep]
            cur_masks_bin = mask_pred[keep].clone()

            cur_prob_masks = cur_scores.view(-1, 1, 1, 1) * cur_masks

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

    # def __call__(self, data):
    #     model_out = self.model(data)
    #     outputs, targets = model_out[1], model_out[3]

    #     self.model_old.eval()
    #     model_out_old = self.model_old(data) if self.use_kd or self.pseudolabeling else None
    #     outputs_old = model_out_old[1] if self.use_kd and model_out_old is not None else None

    #     # prepare targets...
    #     if "instances" in data[0]:
    #         # gt_instances = [x["instances"].to(self.device) for x in data]
    #         # targets = VideoMaskFormer.prepare_targets(gt_instances, model_out['shape'], per_pixel=False)

    #         # Labels assume that background is class 0, remove it.
    #         if not self.use_bg:
    #             for tar in targets:
    #                 tar['labels'] -= 1

    #         # Pseudo-labeling algorithm
    #         if self.pseudolabeling:
    #             targets = self.make_pseudolabels2(outputs_old, data, targets)

    #     else:
    #         targets = None

    #     # bipartite matching-based loss
    #     losses = self.criterion(outputs, targets, outputs_old)

    #     if self.pod_weight > 0:
    #         losses.update(pod_loss(model_out, model_out_old))

    #     for k in list(losses.keys()):
    #         if k in self.criterion.weight_dict:
    #             losses[k] *= self.criterion.weight_dict[k]
    #         else:
    #             # remove this loss if not specified in `weight_dict`
    #             losses.pop(k)

    #     return losses
    def __call__(self, data):
        model_out = self.model(data)
        # outputs, targets = model_out[1], model_out[3] # 原始的
        if isinstance(model_out, tuple):
            losses = model_out[0]
        else:
            # use coMformer
            outputs = model_out['outputs']
            targets = model_out['targets']
            # losses = model_out['losses']
            if self.model_old != None:
                self.model_old.eval()
                model_out_old = self.model_old(data) if self.use_kd or self.pseudolabeling else None
                outputs_old = model_out_old[1] if self.use_kd and model_out_old is not None else None

            # prepare targets...
            if "instances" in data[0]:
                images = []
                for video in data:
                    for frame in video["image"]:
                        images.append(frame.to(self.device))
                pixel_mean = self.model.pixel_mean if not isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model.module.pixel_mean
                pixel_std = self.model.pixel_std if not isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model.module.pixel_std
                images = [(x - pixel_mean) / pixel_std for x in images]
                images = ImageList.from_tensors(images, self.model.size_divisibility)
                # assert targets==None
                num_frames = self.model.num_frames
                targets = self.prepare_targets(num_frames, data, images)
                # Labels assume that background is class 0, remove it.
                if not self.use_bg:
                    for tar in targets:
                        tar['labels'] -= 1

                # Pseudo-labeling algorithm
                if self.pseudolabeling:
                    targets = self.make_pseudolabels2(outputs_old, data, targets)

            else:
                targets = None

            # bipartite matching-based loss
            losses = self.criterion(outputs, targets, outputs_old)

            if self.pod_weight > 0:
                losses.update(pod_loss(model_out, model_out_old))

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)

        return losses
    def prepare_targets(num_frames, targets, images, device='cuda'):
        h_pad, w_pad = images.tensor.shape[-2:]
        gt_instances = []
        for targets_per_video in targets:
            _num_instance = len(targets_per_video["instances"][0])
            mask_shape = [_num_instance, num_frames, h_pad, w_pad]
            gt_masks_per_video = torch.zeros(mask_shape, dtype=torch.bool, device=device)

            gt_ids_per_video = []
            for f_i, targets_per_frame in enumerate(targets_per_video["instances"]):
                targets_per_frame = targets_per_frame.to(device)
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