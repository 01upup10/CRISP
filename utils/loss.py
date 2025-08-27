import torch.nn as nn
import torch.nn.functional as F
import torch


def get_loss(loss_type):
    if loss_type == 'focal_loss':
        return FocalLoss(ignore_index=255, size_average=True)
    elif loss_type == 'cross_entropy':
        return nn.CrossEntropyLoss(ignore_index=255, reduction='mean')


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index=ignore_index
        self.size_average=size_average

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()


class BCEWithLogitsLossWithIgnoreIndex(nn.Module):
    def __init__(self, reduction='mean', ignore_index=255):
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # inputs of size B x C x H x W
        n_cl = torch.tensor(inputs.shape[1]).to(inputs.device)
        labels_new = torch.where(targets != self.ignore_index, targets, n_cl)
        # replace ignore with numclasses + 1 (to enable one hot and then remove it)
        targets = F.one_hot(labels_new, inputs.shape[1] + 1).float().permute(0, 3, 1, 2)
        targets = targets[:, :inputs.shape[1], :, :]  # remove 255 from 1hot
        # targets is B x C x H x W so shape[1] is C
        loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        # loss has shape B x C x H x W
        loss = loss.sum(dim=1)  # sum the contributions of the classes
        if self.reduction == 'mean':
            # if targets have only zeros, we skip them
            return torch.masked_select(loss, targets.sum(dim=1) != 0).mean()
        elif self.reduction == 'sum':
            return torch.masked_select(loss, targets.sum(dim=1) != 0).sum()
        else:
            return loss * targets.sum(dim=1)


class IcarlLoss(nn.Module):
    def __init__(self, reduction='mean', ignore_index=255, bkg=False):
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.bkg = bkg

    def forward(self, inputs, targets, output_old):
        # inputs of size B x C x H x W
        n_cl = torch.tensor(inputs.shape[1]).to(inputs.device)
        labels_new = torch.where(targets != self.ignore_index, targets, n_cl)
        # replace ignore with numclasses + 1 (to enable one hot and then remove it)
        targets = F.one_hot(labels_new, inputs.shape[1] + 1).float().permute(0, 3, 1, 2)
        targets = targets[:, :inputs.shape[1], :, :]  # remove 255 from 1hot
        if self.bkg:
            targets[:, 1:output_old.shape[1], :, :] = output_old[:, 1:, :, :]
        else:
            targets[:, :output_old.shape[1], :, :] = output_old

        # targets is B x C x H x W so shape[1] is C
        loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        # loss has shape B x C x H x W
        loss = loss.sum(dim=1)  # sum the contributions of the classes
        if self.reduction == 'mean':
            # if targets have only zeros, we skip them
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class UnbiasedCrossEntropy(nn.Module):
    def __init__(self, old_cl=None, reduction='mean', ignore_index=255):
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.old_cl = old_cl

    def forward(self, inputs, targets):

        old_cl = self.old_cl
        outputs = torch.zeros_like(inputs)  # B, C (1+V+N), H, W
        den = torch.logsumexp(inputs, dim=1)                               # B, H, W       den of softmax
        outputs[:, 0] = torch.logsumexp(inputs[:, 0:old_cl], dim=1) - den  # B, H, W       p(O)
        outputs[:, old_cl:] = inputs[:, old_cl:] - den.unsqueeze(dim=1)    # B, N, H, W    p(N_i)

        labels = targets.clone()    # B, H, W
        labels[targets < old_cl] = 0  # just to be sure that all labels old belongs to zero

        loss = F.nll_loss(outputs, labels, ignore_index=self.ignore_index, reduction=self.reduction)

        return loss


class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, reduction='mean', alpha=1.):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, inputs, targets, mask=None):
        inputs = inputs.narrow(1, 0, targets.shape[1])

        outputs = torch.log_softmax(inputs, dim=1)
        labels = torch.softmax(targets * self.alpha, dim=1)

        loss = (outputs * labels).mean(dim=1)

        if mask is not None:
            loss = loss * mask.float()

        if self.reduction == 'mean':
            outputs = -torch.mean(loss)
        elif self.reduction == 'sum':
            outputs = -torch.sum(loss)
        else:
            outputs = -loss

        return outputs


class UnbiasedKnowledgeDistillationLoss(nn.Module):
    def __init__(self, reduction='mean', alpha=1., classes=[30], T=1., deep_sup=True):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha
        self.T = T
        # assert isinstance(classes, list) and len(classes) < 2, 'classes not list orlen(classes)<2'
        self.classes = classes
        self.deep_sup = deep_sup

    def forward(self, inputs, targets, mask=None, mode="2-2"):
        if mode=="1":
            '''
            mode==1: return 0.001*(kl_div_student_teacher + mask_kd)
            mode==2-1 return loss_class_kd
            mode==2-2 return loss_class_kd, loss_masks_kd
            '''
            assert isinstance(inputs, dict), "Error unkd_forward inputs are not dict"
            new_cl = self.classes[-1]
            targets_labels = targets['pred_logits']  # n*100 *n_class_old+1 Teacher
            targets_masks = targets['pred_masks'] # n*100*2*H*W

            # new_bkg_idx = torch.tensor([0] + [x for x in range(self.classes[-2], self.classes[-1]+self.classes[-2])]).to(inputs['pred_logits'].device)
            new_bkg_idx = torch.tensor([0] + [x for x in range(sum(self.classes[:-1]), sum(self.classes))]).to(inputs['pred_logits'].device)
            # if inputs['pred_masks'].dim() != 4:
            #     _pred_masks  = inputs['pred_masks']
            #     N, C, H, W = _pred_masks.shape[0]*_pred_masks.shape[1],_pred_masks.shape[2], _pred_masks.shape[3], _pred_masks.shape[4]
            #     _pred_masks = _pred_masks.reshape((N, C, H, W))
            #     pred_masks  = _pred_masks
            # else:
            #     pred_masks = inputs['pred_masks']
            
            pred_labels = inputs['pred_logits'] # n *100 *n_class_old+1 Student
            pred_labels_softmax = pred_labels[:, :-1] # n*99*n_classes_pld+1
            # targets_labels_softmax = F.softmax(targets_labels, dim=-1)[:, :-1]
            if targets_labels.shape[-1] != pred_labels_softmax.shape[-1]:
                targets_labels = torch.nn.functional.pad(targets_labels, (0, self.classes[-1], 0, 0, 0, 0)) # padding to same dim use 0
            targets_labels_softmax = F.softmax(targets_labels, dim=-1)[:, :-1]
            kl_div_student_teacher = F.kl_div(
                F.log_softmax(pred_labels_softmax, dim=-1) / self.T,
                targets_labels_softmax,
                reduction='batchmean'
            )

            pred_masks = inputs['pred_masks']
            pred_masks_softmax = F.softmax(pred_masks, dim=2)
            targets_masks_softmax = F.softmax(targets_masks, dim=2)
            mask_kd = F.cross_entropy(pred_masks_softmax, targets_masks_softmax)

            # KL divergence between teacher and student
            # kl_div_teacher_student = F.kl_div(
            #     F.log_softmax(targets_labels_softmax[:, :, :new_bkg_idx[1]] / self.T, dim=-1),
            #     pred_labels_softmax[:, :, :new_bkg_idx[1]],
            #     reduction='batchmean'
            # )

            # Compute UKD loss
            ukd_loss = kl_div_student_teacher + mask_kd #- kl_div_teacher_student
            return {'lkd':0.001 * (ukd_loss ** self.T)}
            
            # den = torch.logsumexp(pred_labels, dim=1)  
            # outputs_no_bgk = pred_labels[:, 1:-new_cl] - den.unsqueeze(dim=1) 
            # outputs_bkg = torch.logsumexp(torch.index_select(pred_labels, index=new_bkg_idx, dim=1), dim=1) - den
        elif "2" in mode:
            '''
            class kd loss
            '''
            losses_kd = {}
            src_logits = inputs['pred_logits']
            tar_logits = targets['pred_logits'].float()
            if self.ukd:
                loss_class_kd = unbiased_knowledge_distillation_loss(src_logits.transpose(1, 2), tar_logits.transpose(1, 2),
                                                               reweight=self.kd_reweight, temperature=self.alpha)
            elif self.l2:
                loss_class_kd = L2_distillation_loss(src_logits.transpose(1, 2), tar_logits.transpose(1, 2))
            else:
                loss_class_kd = knowledge_distillation_loss(src_logits.transpose(1, 2), tar_logits.transpose(1, 2),
                                                      use_new=self.kd_use_novel, reweight=self.kd_reweight,
                                                      temperature=self.alpha)
            # if "aux_outputs" in inputs and self.deep_sup:
            #     for i, aux_outputs in enumerate(outputs["aux_outputs"]):
            #         indices = self.matcher(aux_outputs, targets)
            #         for loss in self.losses:
            #             if outputs_old is not None and self.kd_deep:
            #                 l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks,
            #                                     outputs_old["aux_outputs"][i])
            #             else:
            #                 l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks)
            #             l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
            #             losses.update(l_dict)
            losses_kd.update({'loss_class_kd':loss_class_kd})
            if mode=="2-2:": 
                new_masks = inputs["pred_masks"]
                old_masks = targets["pred_masks"].detach()
                labels_masks = old_masks.sigmoid()
                loss_masks_kd = F.binary_cross_entropy_with_logits(new_masks, labels_masks)
                losses_kd.update({'loss_masks_kd':loss_masks_kd})
            
            return losses_kd

    
        else:

            new_cl = inputs.shape[1] - targets.shape[1]

            targets = targets * self.alpha

            new_bkg_idx = torch.tensor([0] + [x for x in range(targets.shape[1], inputs.shape[1])]).to(inputs.device)

            den = torch.logsumexp(inputs, dim=1)                          # B, H, W
            outputs_no_bgk = inputs[:, 1:-new_cl] - den.unsqueeze(dim=1)  # B, OLD_CL, H, W
            outputs_bkg = torch.logsumexp(torch.index_select(inputs, index=new_bkg_idx, dim=1), dim=1) - den     # B, H, W

            labels = torch.softmax(targets, dim=1)                        # B, BKG + OLD_CL, H, W

            # make the average on the classes 1/n_cl \sum{c=1..n_cl} L_c
            loss = (labels[:, 0] * outputs_bkg + (labels[:, 1:] * outputs_no_bgk).sum(dim=1)) / targets.shape[1]

            if mask is not None:
                loss = loss * mask.float()

            if self.reduction == 'mean':
                    outputs = -torch.mean(loss)
            elif self.reduction == 'sum':
                    outputs = -torch.sum(loss)
            else:
                outputs = -loss

            targets_labels = targets[0]['labels']  # n *100 *n_class_old
        return outputs
    
    
def focal_loss(inputs, targets, alpha=10, gamma=2, reduction='mean', ignore_index=255):
    ce_loss = F.cross_entropy(inputs, targets, reduction="none", ignore_index=ignore_index)
    pt = torch.exp(-ce_loss)
    f_loss = alpha * (1 - pt) ** gamma * ce_loss
    if reduction == 'mean':
        f_loss = f_loss.mean()
    elif reduction == 'wmean':
        f_loss = f_loss.sum() / ((1 - pt) ** gamma).sum()
    return f_loss


def focal_uce_loss(inputs, targets, old_cl, alpha=10, gamma=2, reduction='mean'):
    ce_loss = unbiased_cross_entropy_loss(inputs, targets, reduction="none", old_cl=old_cl)
    pt = torch.exp(-ce_loss)
    f_loss = alpha * (1 - pt) ** gamma * ce_loss
    if reduction == 'mean':
        f_loss = f_loss.mean()
    elif reduction == 'wmean':
        f_loss = f_loss.sum() / ((1 - pt) ** gamma).sum()
    return f_loss


def unbiased_cross_entropy_loss(inputs, targets, old_cl, weights=None, reduction='mean'):
    outputs = torch.zeros_like(inputs)  # B, C (1+V+N), H, W
    den = torch.logsumexp(inputs, dim=1)  # B, H, W       den of softmax

    to_sum = torch.cat((inputs[:, -1:], inputs[:, 0:old_cl]), dim=1)
    outputs[:, -1] = torch.logsumexp(to_sum, dim=1) - den  # B, K       p(O)
    outputs[:, :-1] = inputs[:, :-1] - den.unsqueeze(dim=1)  # B, N, K    p(N_i)

    loss = F.nll_loss(outputs, targets, weight=weights, reduction=reduction)

    return loss


def focal_distillation_loss(inputs, targets, use_new=False, alpha=1, gamma=2, ):
    if use_new:
        outputs = torch.log_softmax(inputs, dim=1)  # remove no-class
        outputs = torch.cat((outputs[:, :targets.shape[1] - 1], outputs[:, -1:]), dim=1)  # only old classes or EOS
    else:
        inputs = torch.cat((inputs[:, :targets.shape[1] - 1], inputs[:, -1:]), dim=1)  # only old classes or EOS
        outputs = torch.log_softmax(inputs, dim=1)
    labels = torch.softmax(targets * alpha, dim=1)
    labels_log = torch.log_softmax(targets * alpha, dim=1)

    loss = (labels * (labels_log-outputs)).sum(dim=1)
    pt = torch.exp(torch.clamp(-loss, max=0))
    f_loss = alpha * (1 - pt) ** gamma * loss

    return f_loss.mean()


def L2_distillation_loss(inputs, targets, use_new=False):
    inputs = torch.cat((inputs[:, :targets.shape[1]-1], inputs[:, -1:]), dim=1)  # only old classes or EOS

    labels = torch.softmax(targets, dim=1)
    outputs = torch.softmax(inputs, dim=1)

    # keep only the informative ones -> The not no-obj masks
    # keep = (labels.argmax(dim=1) != targets.shape[1]-1)  # B x Q

    loss = torch.pow((outputs - labels), 2).sum(dim=1)
    # loss = (loss * keep).sum() / (keep.sum() + 1e-4)  # keep only obj queries, 1e-4 to avoid NaN
    return loss.mean()


def knowledge_distillation_loss(inputs, targets, reweight=False, gamma=2., temperature=1., use_new=True):
    if use_new:
        outputs = torch.log_softmax(inputs, dim=1)  # remove no-class
        outputs = torch.cat((outputs[:, :targets.shape[1]-1], outputs[:, -1:]), dim=1)  # only old classes or EOS
    else:
        inputs = torch.cat((inputs[:, :targets.shape[1]-1], inputs[:, -1:]), dim=1)  # only old classes or EOS
        outputs = torch.log_softmax(inputs, dim=1)
    labels = torch.softmax(targets * temperature, dim=1)
    labels_log = torch.log_softmax(targets * temperature, dim=1)

    loss = (labels*(labels_log - outputs)).sum(dim=1)  # B x Q
    # Re-weight no-cls queries as in classification
    if reweight:
        loss = ((1-labels[:, -1]) ** gamma * loss).sum() / ((1-labels[:, -1]) ** gamma).sum()
    else:
        loss = loss.mean()
    return loss


def unbiased_knowledge_distillation_loss(inputs, targets, reweight=False, gamma=2., temperature=1.):
    '''
    inputs: (n_frames, C, Q)
    targets:(n_frames, C, Q)

    '''
    targets = targets * temperature

    den = torch.logsumexp(inputs, dim=1)  # n_frames, C, Q
    outputs_no_bgk = inputs[:, :targets.shape[1]-1] - den.unsqueeze(dim=1)  # B, OLD_CL, Q
    outputs_bkg = torch.logsumexp(inputs[:, targets.shape[1]-1:], dim=1) - den  # B, Q
    labels = torch.softmax(targets, dim=1)  # B, BKG + OLD_CL, Q
    labels_soft = torch.log_softmax(targets, dim=1)

    loss = labels[:, -1] * (labels_soft[:, -1] - outputs_bkg) + \
           (labels[:, :-1] * (labels_soft[:, :-1] - outputs_no_bgk)).sum(dim=1)  # B, Q
    # Re-weight no-cls queries as in classificaton
    if reweight:
        loss = ((1-labels[:, -1]) ** gamma * loss).sum() / ((1-labels[:, -1]) ** gamma).sum()
    else:
        loss = loss.mean()
    return loss