# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
import logging
import fvcore.nn.weight_init as weight_init
from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d

from mask2former.modeling.transformer_decoder.maskformer_transformer_decoder import TRANSFORMER_DECODER_REGISTRY

from .position_encoding import PositionEmbeddingSine3D
from typing import Optional, List
from continual.modeling import IncrementalClassifier, CosineClassifier, IncrementalLinearClassifier
from copy import deepcopy
import numpy as np

import einops
from .crisp_prompter import CLIP_Prompter, CLIP_Prompter_WithoutEncoder
import tracemalloc
from detectron2.data import MetadataCatalog
# classes_names = [
#     'person', 'giant_panda', 'lizard', 'parrot', 'skateboard', 'sedan', 
#     'ape', 'dog', 'snake', 'monkey', 'hand', 'rabbit', 'duck', 'cat', 
#     'cow', 'fish', 'train', 'horse', 'turtle', 'bear', 'motorbike', 
#     'giraffe', 'leopard', 'fox', 'deer', 'owl', 'surfboard', 'airplane', 
#     'truck', 'zebra', 'tiger', 'elephant', 'snowboard', 'boat', 'shark', 
#     'mouse', 'frog', 'eagle', 'earless_seal', 'tennis_racket'
# ]

def sigmoid_to_logit(x):
    x = x.clamp(0.001, 0.999)
    return torch.log(x / (1-x))

class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False, residual_prompts=None):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     residual_prompts:  Optional[Tensor] = None, ):
        q = k = self.with_pos_embed(tgt, query_pos)
        # tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
        #                       key_padding_mask=tgt_key_padding_mask)[0]
        if residual_prompts is not None:
            tgt2 = self.self_attn(q, k, value=tgt+residual_prompts, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        else:
            tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2) # if residual_prompts is None else tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        
        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None, residual_prompts: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos, residual_prompts)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        # tgt2, att_map = tgt2[0], tgt2[1]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        
        # return (tgt, att_map)
        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


@TRANSFORMER_DECODER_REGISTRY.register()
class VideoMultiScaleMaskedTransformerDecoder(nn.Module):

    _version = 2

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            # Do not warn if train from scratch
            scratch = True
            logger = logging.getLogger(__name__)
            for k in list(state_dict.keys()):
                newk = k
                if "static_query" in k:
                    newk = k.replace("static_query", "query_feat")
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]
                    scratch = False

            if not scratch:
                logger.warning(
                    f"Weight format of {self.__class__.__name__} have changed! "
                    "Please upgrade your models. Applying automatic conversion now ..."
                )

    @configurable
    def __init__(
        self,
        in_channels,
        mask_classification=True,
        *,
        num_classes: int,
        hidden_dim: int,
        num_queries: int,
        nheads: int,
        dim_feedforward: int,
        dec_layers: int,
        pre_norm: bool,
        mask_dim: int,
        enforce_input_project: bool,
        # from ECLIPSE
        num_prompts: int,
        prompt_deep: bool = False,
        softmask: bool = False,
        inc_query: Optional[bool] = None,
        cosine: Optional[bool] = False,
        bias: Optional[bool] = False,
        classes: Optional[List[int]] = None,
        prompt_mask_mlp: Optional[bool] = False,
        prompt_no_obj_mlp: Optional[bool] = False,
        deep_cls: Optional[bool] = False,
        deltas: Optional[List[float]] = None,
        # video related
        num_frames,
        # is valid
        is_valid = False,
        use_appearance_decoder: Optional[bool] = False,
        use_clip_prompter: Optional[bool] = False,
        without_text_encoder: Optional[bool] = False,
        infer_with_clip_prompter: Optional[bool] = False,
        use_contrastive_loss: Optional[bool] = False,
        use_orth_loss: Optional[bool] = False,
        orth_layers: Optional[int] = 10,
        datasets_name: Optional[str] = None
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification

        self.num_frames = num_frames
        self.softmask = softmask # from ECLIPSE
        self.inc_query = inc_query # from ECLIPSE
        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine3D(N_steps, normalize=True)
        
        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        # output FFNs
        if self.mask_classification:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

        self.is_valid = is_valid
        self.num_classes = num_classes

        # Parameters for ECLIPSE
        self.num_prompts = num_prompts # number of prompts
        self.prompt_deep = prompt_deep and self.num_prompts > 0 
        self.prompt_mask_mlp = prompt_mask_mlp and self.num_prompts > 0
        self.prompt_no_obj_mlp = prompt_no_obj_mlp and self.num_prompts > 0

        self.deltas = deltas
        if self.deltas is None:
            self.deltas = [0.0 for _ in classes]
        elif type(self.deltas) == float:
            self.deltas = [self.deltas for _ in classes]
        elif len(self.deltas) > len(classes):
            self.deltas = self.deltas[:len(classes)]
        elif len(self.deltas) < len(classes):
            self.deltas = self.deltas + [self.deltas[-1] for _ in range(len(classes)-len(self.deltas))]
            
        assert len(self.deltas) == len(classes), "CONT."
        
        self.old_model = False
        
        # prompt embeddings
        if self.num_prompts > 0:
            self.prompt_feat = nn.ModuleList(
                [nn.Embedding(num_prompts, hidden_dim) for _ in classes[1:]]
            )
            
            if self.prompt_deep:
                self.prompt_embed = nn.ModuleList(
                    [
                        nn.ModuleList(
                            [nn.Embedding(num_prompts, hidden_dim) for _ in range(self.num_layers)]
                        ) for _ in classes[1:]
                    ]
                )
                
            else:
                self.prompt_embed = nn.ModuleList(
                    [nn.Embedding(num_prompts, hidden_dim) for _ in classes[1:]]
                )

        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        # output FFNs
        self.num_classes = num_classes
        self.classes = classes # [C0, C1, ..., Cn]
        
        if self.mask_classification:
            if classes is not None:
                if cosine:
                    self.class_embed = CosineClassifier([1] + classes, channels=hidden_dim)
                else:
                    # [1] : no_obj, (we don't have bkg class)
                    self.class_embed = IncrementalClassifier(
                        [1] + classes, 
                        channels=hidden_dim, 
                        bias=bias, 
                        deep_cls=deep_cls,
                    )
            else:
                self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
                
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
        
        # self.use_prompt_adapter = use_prompt_adapter
        # if self.use_prompt_adapter:
        #     self.prompt_adapter = PromptAdapter([1] + classes) # prompt_adapter
            # self.prompt_embed_adapter = PromptAdapter([1] + classes) 
        # self.base_query_embed = deepcopy(self.query_embed.weight.data)
        self.base_query_feat = deepcopy(self.query_feat.weight.data)

        if self.prompt_mask_mlp:
            self.prompt_mask_embed = nn.ModuleList(
                [deepcopy(self.mask_embed) for _ in classes[1:]]
            )
            
        if self.prompt_no_obj_mlp:
            self.prompt_no_obj_embed = nn.ModuleList(
                [MLP(hidden_dim, hidden_dim, 1, 3) for _ in classes[1:]]
            )

        self.use_clip_prompter = use_clip_prompter
        self.without_text_encoder = without_text_encoder
        self.infer_with_clip_prompter = infer_with_clip_prompter
        self.use_contrastive_loss = use_contrastive_loss
        self.use_orth_loss = use_orth_loss
        self.orth_layers = orth_layers
        if self.use_clip_prompter:
            classes_names = MetadataCatalog.get(datasets_name).thing_classes
            self.sem_prompter = CLIP_Prompter(classes, len(classes)-1, classes_names, prompt_dim=512, clip_model='ViT-B/32', device='cuda')
            if self.without_text_encoder:
                self.sem_prompter = CLIP_Prompter_WithoutEncoder(classes, len(classes)-1, classes_names, prompt_dim=512, clip_model='ViT-B/32', device='cuda')
    def set_as_old_model(self, ):
        self.old_model = True
        self.prompt_feat = None
        self.prompt_embed = None
        self.prompt_mask_mlp = False
        self.prompt_no_obj_mlp = False

    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification, old_params=None):
        ret = {}
        ret["in_channels"] = in_channels
        ret["mask_classification"] = mask_classification
        ret["softmask"] = cfg.MODEL.MASK_FORMER.SOFTMASK

        if hasattr(cfg, "CONT"):
            ret['inc_query'] = cfg.CONT.INC_QUERY
            ret["classes"] = [cfg.CONT.BASE_CLS] + cfg.CONT.TASK*[cfg.CONT.INC_CLS]
            ret["num_classes"] = cfg.CONT.BASE_CLS + cfg.CONT.TASK * cfg.CONT.INC_CLS
            ret["cosine"] = cfg.CONT.COSINE
            ret["bias"] = cfg.CONT.USE_BIAS
            if cfg.MODEL.MASK_FORMER.TEST.MASK_BG and (cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON or
                                                       cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON):
                ret["num_classes"] += 1
                ret["classes"][0] += 1
            
            # Parameters for ECLIPSE
            ret['num_prompts'] = cfg.CONT.NUM_PROMPTS
            ret['prompt_deep'] = cfg.CONT.PROMPT_DEEP
            ret['prompt_mask_mlp'] = cfg.CONT.PROMPT_MASK_MLP
            ret['prompt_no_obj_mlp'] = cfg.CONT.PROMPT_NO_OBJ_MLP
            ret['deltas'] = cfg.CONT.LOGIT_MANI_DELTAS
            ret['deep_cls'] = cfg.CONT.DEEP_CLS

        else:
            ret['inc_query'] = None
            ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
            if not cfg.MODEL.MASK_FORMER.TEST.MASK_BG and (cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON or
                                                       cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON):
                ret["num_classes"] -= 1
        ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        ret["num_queries"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        # Transformer parameters:
        ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
        ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD

        # NOTE: because we add learnable query features which requires supervision,
        # we add minus 1 to decoder layers to be consistent with our loss
        # implementation: that is, number of auxiliary losses is always
        # equal to number of decoder layers. With learnable query features, the number of
        # auxiliary losses equals number of decoders plus 1.
        assert cfg.MODEL.MASK_FORMER.DEC_LAYERS >= 1
        ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1
        ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
        ret["enforce_input_project"] = cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ

        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM

        ret['num_frames'] = cfg.INPUT.SAMPLING_FRAME_NUM
        
        # <<<crisp>>>
        ret["use_clip_prompter"] = cfg.CRISP.USE_CLIP_PROMPTER
        ret["without_text_encoder"] = cfg.CRISP.WITHOUT_TEXT_ENCODER
        ret["infer_with_clip_prompter"] = cfg.CRISP.INFER_WITH_CLIP_PROMPTER
        ret['use_contrastive_loss'] = cfg.CRISP.USE_CONTRASTIVE_LOSS 
        ret['use_orth_loss'] = cfg.CRISP.USE_ORTH_LOSS
        ret['orth_layers'] = cfg.CRISP.ORTH_LAYERS
        ret['datasets_name'] = cfg.DATASETS.TRAIN[0]

        return ret

    
    def forward(self, x, mask_features, mask=None):
    
        bt, c_m, h_m, w_m = mask_features.shape
        bs = bt // self.num_frames if self.training or self.is_valid else 1
        t = bt // bs

        # mask_features = mask_features.view(bs, t, c_m, h_m, w_m)

        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        # disable mask, it does not affect performance
        del mask

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i].view(bs, t, -1, size_list[-1][0], size_list[-1][1]), None).flatten(3))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # NTxCxHW => NxTxCxHW => (TxHW)xNxC
            _, c, hw = src[-1].shape
            pos[-1] = pos[-1].view(bs, t, c, hw).permute(1, 3, 0, 2).flatten(0, 1)
            src[-1] = src[-1].view(bs, t, c, hw).permute(1, 3, 0, 2).flatten(0, 1)

        if not self.training:
            if not self.infer_with_clip_prompter:
                return self.forward_infer(x, src, pos, size_list, mask_features)
            else:
                return self.forward_infer_with_clip_prompts(x, src, pos, size_list, mask_features)
        else:
            if self.num_prompts > 0 and not self.old_model:
                return self.forward_new_train(x, src, pos, size_list, mask_features)
            else:
                return self.forward_base_train(x, src, pos, size_list, mask_features)


    def forward_prediction_heads(self, output, class_embed, mask_embed, mask_features, 
                                 attn_mask_target_size, qdims=None):
        
        decoder_output = self.decoder_norm(output) 
        decoder_output = decoder_output.transpose(0, 1)
        
        outputs_class = class_embed(decoder_output) # outputs_class : [B, 100, 100+10+1]
        
        # logit manipulation implementation
        if not self.training and self.num_prompts > 0 and qdims is not None:
            m_embed = []
            for n in range(len(qdims)-1):
                m_embed.append(mask_embed[n](decoder_output[:, qdims[n]:qdims[n+1]]))
                
                if self.prompt_no_obj_mlp and n > 0:
                    no_obj_logit = self.prompt_no_obj_embed[n-1](decoder_output)
                    outputs_class[:, qdims[n]:qdims[n+1], -1] = no_obj_logit[:, qdims[n]:qdims[n+1], 0]
                    
                if self.deltas[n] > 0:
                    # logit manipulation with delta: aggregation of other class knowledge
                    noobj_score = outputs_class[:, qdims[n]:qdims[n+1], 
                                                list(range(0, sum(self.classes[:n]))) + \
                                                list(range(sum(self.classes[:n+1]), sum(self.classes)))
                                               ].sigmoid().sum(2).clamp(0., 1.)

                    outputs_class[:, qdims[n]:qdims[n+1], -1] = sigmoid_to_logit(
                        noobj_score * self.deltas[n]
                    )
                    
                elif self.deltas[n] < 0:
                    # negative delta means calibration the class logits without aggregation of other class knowledge
                    # we empirically found that this strategy is effective when the number of incremental steps is small (e.g., 100-50).
                    outputs_class[:, qdims[n]:qdims[n+1], -1] = sigmoid_to_logit(
                        outputs_class[:, qdims[n]:qdims[n+1], -1].sigmoid() * -self.deltas[n]
                    )
                        
                # deactivate other class logits: regarding sigmoid(-10) => 0.0
                outputs_class[:, qdims[n]:qdims[n+1], 
                              list(range(0, sum(self.classes[:n]))) + \
                              list(range(sum(self.classes[:n+1]), sum(self.classes)))
                             ] = -10
                    
            m_embed = torch.cat(m_embed, dim=1)
            
        else:
            m_embed = mask_embed(decoder_output)
            if self.prompt_no_obj_mlp and qdims is not None:
                for n in range(1, len(qdims)-1):
                    no_obj_logit = self.prompt_no_obj_embed[n-1](decoder_output)
                    outputs_class[:, qdims[n]:qdims[n+1], -1] = no_obj_logit[:, qdims[n]:qdims[n+1], 0]
            
        # outputs_masks = torch.einsum("bqc,bchw->bqhw", m_embed, mask_features)
        outputs_masks = torch.einsum("bqc,btchw->bqthw", m_embed, mask_features)
            
        # # NOTE: prediction is of higher-resolution
        # # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        # attn_mask = F.interpolate(outputs_masks, size=attn_mask_target_size, mode="bilinear", align_corners=False)

        # NOTE: prediction is of higher-resolution
        # [B, Q, T, H, W] -> [B, Q, T*H*W] -> [B, h, Q, T*H*W] -> [B*h, Q, T*HW]
        b, q, t, _, _ = outputs_masks.shape
        attn_mask = F.interpolate(outputs_masks.flatten(0, 1), size=attn_mask_target_size, mode="bilinear", align_corners=False).view(
            b, q, t, attn_mask_target_size[0], attn_mask_target_size[1])

        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        if self.softmask:
            attn_mask = (attn_mask.softmax(dim=1).flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
        else:
            attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()

        return outputs_class, outputs_masks, attn_mask

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
            ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]
        
    def forward_new_train(self, x, src, pos, size_list, mask_features):
        """
        Training the model for novel classes (applying visual prompt tuning, step > 1)
        """
        # visage
        # if self.use_appearance_decoder:
        #     _, bs, _ = src[0].shape
        bt, c_m, h_m, w_m = mask_features.shape
        bs = bt // self.num_frames if self.training or self.is_valid else 1 # batch_size, num_frames=2(training)
        t = bt // bs
        mask_features = mask_features.view(bs, t, c_m, h_m, w_m)
        predictions_class = []
        predictions_mask = []
        

        if self.num_prompts > 0 and self.prompt_no_obj_mlp:
            query_dims = np.cumsum([0, self.num_queries] + [self.num_prompts] * len(self.prompt_embed))
        else:
            query_dims = None
            
        output_p = self.prompt_feat[-1].weight.unsqueeze(1).repeat(1, bs, 1)

        # prediction heads on learnable query features
        outputs_class_p, outputs_mask_p, attn_mask_p = self.forward_prediction_heads(
            output_p, 
            self.class_embed, 
            self.mask_embed if not self.prompt_mask_mlp else self.prompt_mask_embed[-1],
            mask_features, 
            attn_mask_target_size=size_list[0],
            qdims=query_dims,
        )
        # crisp prompts
        if self.use_clip_prompter:
            prompts = self.sem_prompter.get_prompts() # .unsqueeze(1).repeat(1, bs, 1)
            prompts, _, loss_contrastive = self.query_matching(prompts, self.prompt_feat[-1].weight)
            residual_prompts = prompts.unsqueeze(1).repeat(1, bs, 1)
        else:
            residual_prompts = None

        predictions_class.append(outputs_class_p)
        predictions_mask.append(outputs_mask_p)

        med_token = [output_p]

        for i in range(self.num_layers):
            if self.prompt_deep:
                prompt_embed = self.prompt_embed[-1][i].weight.unsqueeze(1).repeat(1, bs, 1)
                # cat原始prompt
                # prompt_embed = torch.cat([query_embed, prompt_embed], dim=0)
            else:
                prompt_embed = self.prompt_embed[-1].weight.unsqueeze(1).repeat(1, bs, 1)

            level_index = i % self.num_feature_levels
            attn_mask_p[torch.where(attn_mask_p.sum(-1) == attn_mask_p.shape[-1])] = False

            # attention: cross-attention first
            output_p = self.transformer_cross_attention_layers[i](
                output_p, 
                src[level_index],
                memory_mask=attn_mask_p,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=prompt_embed,
            )

            output_p = self.transformer_self_attention_layers[i](
                output_p, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=prompt_embed,
                residual_prompts=residual_prompts
            )

            # FFN
            output_p = self.transformer_ffn_layers[i](
                output_p
            )
            med_token.append(output_p)

            outputs_class_p, outputs_mask_p, attn_mask_p = self.forward_prediction_heads(
                output_p, 
                self.class_embed, 
                self.mask_embed if not self.prompt_mask_mlp else self.prompt_mask_embed[-1],
                mask_features, 
                attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels],
                qdims=query_dims,
            )

        predictions_class.append(outputs_class_p)
        predictions_mask.append(outputs_mask_p)
    
        # pred_embds = self.decoder_norm(output_p)
        # pred_embds = einops.rearrange(pred_embds, 'q (b t) c -> b t q c', t=t)
        orth_target = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
        loss_orth = self.calculate_orthogonal_loss(med_token, orth_target)

        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_class if self.mask_classification else None, predictions_mask
            ),
            'loss_contrastive': loss_contrastive if self.use_contrastive_loss else None,
            'loss_orth': loss_orth if self.use_orth_loss else None,
            # 'pred_embds': pred_embds,
            # 'query': query_feat,
            # 'features': mask_features
        }
        return out

    def forward_base_train(self, x, src, pos, size_list, mask_features):
        """
        Training the model for base classes (before applying visual prompt tuning, step 0)
        """
        # tracemalloc.start()
        bt, c_m, h_m, w_m = mask_features.shape
        bs = bt // self.num_frames if self.training or self.is_valid else 1
        t = bt // bs
        mask_features = mask_features.view(bs, t, c_m, h_m, w_m)
        
        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
        # prompts
        if self.use_clip_prompter:
            prompts = self.sem_prompter.get_prompts() # .unsqueeze(1).repeat(1, bs, 1)
            prompts,_,loss_contrastive = self.query_matching(prompts, self.query_feat.weight)
            residual_prompts = prompts.unsqueeze(1).repeat(1, bs, 1)
        else:
            residual_prompts = None
        predictions_class = []
        predictions_mask = []

        # prediction heads on learnable query features
        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
            output, self.class_embed, self.mask_embed, mask_features, 
            attn_mask_target_size=size_list[0], 
        )
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        med_token = [output]

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output, 
                src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=query_embed
            )

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed,
                residual_prompts=residual_prompts, # crisp prompts
            )

            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )

            med_token.append(output)

            # prediction heads on learnable query features
            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
                output, self.class_embed, self.mask_embed, mask_features, 
                attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels],
                )
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

            

        assert len(predictions_class) == self.num_layers + 1
        
        loss_orth = self.calculate_orthogonal_loss(med_token)
        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_class if self.mask_classification else None, predictions_mask
            ),
            'decoder_output': self.decoder_norm(output).transpose(0, 1),
            'loss_contrastive': loss_contrastive if self.use_contrastive_loss else None,
            'loss_orth': loss_orth if self.use_orth_loss else None,
        }

        return out
    
    def forward_infer(self, x, src, pos, size_list, mask_features):
        """
        Inference for ECLIPSE
        """
        bt, c_m, h_m, w_m = mask_features.shape
        bs = bt // self.num_frames if self.training or self.is_valid else 1
        t = bt // bs
        mask_features = mask_features.view(bs, t, c_m, h_m, w_m)
        
        predictions_class = []
        predictions_mask = []
        
        if self.num_prompts > 0:
            mask_embeds = nn.ModuleList([self.mask_embed])
            # if self.prompt_mask_mlp:
            mask_embeds = mask_embeds.extend(self.prompt_mask_embed)
            query_dims = np.cumsum([0, self.num_queries] + [self.num_prompts] * len(self.prompt_embed))
        else:
            mask_embeds = self.mask_embed
            query_dims = None

        # QxNxC
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
        
        if self.num_prompts > 0:
            output = torch.cat(
                [
                    output, 
                    torch.cat([p.weight.unsqueeze(1).repeat(1, bs, 1) for p in self.prompt_feat], dim=0)
                ], dim=0
            )
        
        # prediction heads on learnable query features
        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
            output, 
            self.class_embed, 
            mask_embeds,
            mask_features, 
            attn_mask_target_size=size_list[0], 
            qdims=query_dims,
        )
        
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        for i in range(self.num_layers):
            query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)

            if self.num_prompts > 0:
                if self.prompt_deep:
                    query_embed = torch.cat(
                        [
                            query_embed, 
                            torch.cat([p[i].weight.unsqueeze(1).repeat(1, bs, 1) for p in self.prompt_embed], dim=0)
                        ], dim=0
                    )
                else:
                    query_embed = torch.cat(
                        [
                            query_embed, 
                            torch.cat([p.weight.unsqueeze(1).repeat(1, bs, 1) for p in self.prompt_embed], dim=0)
                        ], dim=0
                    )
            
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output, 
                src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=query_embed
            )
            att_map = None
            if isinstance(output, tuple):
                output, att_map = output[0], output[1]

            if self.num_prompts > 0:
                self_attn_outputs = torch.zeros_like(output)
                for qn, qdim in enumerate(query_dims[:-1]):
                    self_attn_outputs[query_dims[qn]:query_dims[qn+1]] = self.transformer_self_attention_layers[i](
                        output[query_dims[qn]:query_dims[qn+1]], tgt_mask=None,
                        tgt_key_padding_mask=None,
                        query_pos=query_embed[query_dims[qn]:query_dims[qn+1]]
                    )
                output = self_attn_outputs
                
            else:
                output = self.transformer_self_attention_layers[i](
                        output, tgt_mask=None,
                        tgt_key_padding_mask=None,
                        query_pos=query_embed,
                    )
            
            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )
            decoder_feat = output
            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
                output, self.class_embed, mask_embeds, mask_features, 
                attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels],
                qdims=query_dims,
            )
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)
        h, w = x[-1].shape[-2], x[-1].shape[-1]
        att_map = att_map.reshape(bs, -1, bt, h, w) if att_map is not None else None
        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'decoder_feat': decoder_feat,
            'att_map': att_map
        }
        return out
    def forward_infer_with_clip_prompts(self, x, src, pos, size_list, mask_features):
        """
        Inference for ECLIPSE
        """
        bt, c_m, h_m, w_m = mask_features.shape
        bs = bt // self.num_frames if self.training else 1
        t = bt // bs
        mask_features = mask_features.view(bs, t, c_m, h_m, w_m)
        # get crisp prompts
        prompts = self.sem_prompter.get_prompts(train=False)
        
        query_recoder = []
        prompts_recoder = []
        for i, n in enumerate(self.classes):
            if i == 0:
                cur_queries = self.query_feat.weight.data
            else:
                cur_queries = self.prompt_feat[i-1].weight.data
            query_recoder.append(cur_queries)
            cur_prompts = self.query_matching(prompts[i], cur_queries)[0]
            prompts_recoder.append(cur_prompts)

        prompts = torch.cat(prompts_recoder)
        residual_prompts = prompts.unsqueeze(1).repeat(1, bs, 1)

        predictions_class = []
        predictions_mask = []
        
        if self.num_prompts > 0:
            mask_embeds = nn.ModuleList([self.mask_embed])
   
            mask_embeds = mask_embeds.extend(self.prompt_mask_embed)
            query_dims = np.cumsum([0, self.num_queries] + [self.num_prompts] * len(self.prompt_embed))
        else:
            mask_embeds = self.mask_embed
            query_dims = None

        # QxNxC
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
        
        if self.num_prompts > 0:
            output = torch.cat(
                [
                    output, 
                    torch.cat([p.weight.unsqueeze(1).repeat(1, bs, 1) for p in self.prompt_feat], dim=0)
                ], dim=0
            )
        
        # prediction heads on learnable query features
        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
            output, 
            self.class_embed, 
            mask_embeds,
            mask_features, 
            attn_mask_target_size=size_list[0], 
            qdims=query_dims,
        )
        
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        for i in range(self.num_layers):
            query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)

            if self.num_prompts > 0:
                if self.prompt_deep:
                    query_embed = torch.cat(
                        [
                            query_embed, 
                            torch.cat([p[i].weight.unsqueeze(1).repeat(1, bs, 1) for p in self.prompt_embed], dim=0)
                        ], dim=0
                    )
                else:
                    query_embed = torch.cat(
                        [
                            query_embed, 
                            torch.cat([p.weight.unsqueeze(1).repeat(1, bs, 1) for p in self.prompt_embed], dim=0)
                        ], dim=0
                    )
            
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output, 
                src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=query_embed
            )
            output, att_map = output[0], output[1]
            if self.num_prompts > 0:
                self_attn_outputs = torch.zeros_like(output)
                for qn, qdim in enumerate(query_dims[:-1]):
                    self_attn_outputs[query_dims[qn]:query_dims[qn+1]] = self.transformer_self_attention_layers[i](
                        output[query_dims[qn]:query_dims[qn+1]], tgt_mask=None,
                        tgt_key_padding_mask=None,
                        query_pos=query_embed[query_dims[qn]:query_dims[qn+1]],
                        residual_prompts=residual_prompts[query_dims[qn]:query_dims[qn+1]]
                    )
                output = self_attn_outputs
                
            else:
                output = self.transformer_self_attention_layers[i](
                        output, tgt_mask=None,
                        tgt_key_padding_mask=None,
                        query_pos=query_embed,
                        residual_prompts=residual_prompts
                    )
            
            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )
            decoder_feat = output
            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
                output, self.class_embed, mask_embeds, mask_features, 
                attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels],
                qdims=query_dims,
            )
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)
        h, w = x[-1].shape[-2], x[-1].shape[-1]
        att_map = att_map.reshape(bs, -1, bt, h, w) 
        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'decoder_feat': decoder_feat,
            'att_map': att_map
        }

        return out
    
    def query_matching(self, clip_prompts, query):
        """
        Matches queries to prompts using cosine similarity and computes a contrastive loss.
        Args:
            clip_prompts (torch.Tensor): Tensor of shape [num_classes, d_clip=512], representing class prompts.
            query (torch.Tensor): Tensor of shape [num_queries, d=512], representing input queries.
        Returns:
            new_prompts (torch.Tensor): Selected prompts based on matching, shape [num_queries, d].
            indices (torch.Tensor): Indices of the best matching prompts for each query, shape [num_queries].
            loss_contrastive (torch.Tensor): Scalar tensor representing the computed contrastive loss.
        """
        def cosine_similarity(A, B):
            """
            Computes cosine similarity between two sets of vectors.
            Args:
                A (torch.Tensor): Shape [n, d]
                B (torch.Tensor): Shape [m, d]
            Returns:
                torch.Tensor: Cosine similarity matrix of shape [n, m]
            """
            # L2 normalize A and B along the feature dimension
            A_normalized = A / torch.norm(A, p=2, dim=1, keepdim=True).clamp(min=1e-8)
            B_normalized = B / torch.norm(B, p=2, dim=1, keepdim=True).clamp(min=1e-8)
            return torch.mm(A_normalized, B_normalized.t())
        # Compute cosine similarity and get matching probabilities
        prob = torch.softmax(cosine_similarity(query, clip_prompts), dim=-1)  # [num_queries, num_classes]
        # Get the index of the highest probability class for each query
        indices = torch.argmax(prob, dim=-1)  # Shape: [num_queries]
        # Retrieve the corresponding prompts for each query
        new_prompts = clip_prompts[indices]  # Shape: [num_queries, d]
        def contrastive_loss(clip_prompts, query, indices):
            """
            Computes a contrastive loss based on positive and negative sample similarities.
            Args:
                clip_prompts (torch.Tensor): Prompt tensor, shape [num_prompts, D] or [num_prompts, B, D]
                query (torch.Tensor): Query tensor, shape [n_query, D] or [n_query, B, D]
                indices (torch.Tensor): Indices of positive samples, shape [B,] or [1,]
            Returns:
                torch.Tensor: Scalar tensor representing the contrastive loss.
            """
            if clip_prompts.ndim == 2:
                # 2D case: standard matrix multiplication
                sims = query @ clip_prompts.T  # Similarity matrix: [n_query, num_prompts]
                n_query = query.size(0)
                # Extract positive similarities and exponentiate
                pos = torch.exp(sims[torch.arange(n_query), indices])  # [n_query,]
                # Create mask to exclude positive samples
                mask = torch.ones_like(sims, device=sims.device)  # [n_query, num_prompts]
                mask[torch.arange(n_query), indices] = 0
                # Sum of exponentiated negative similarities
                neg = (torch.exp(sims) * mask).sum(dim=1)  # [n_query,]
                # Compute loss per query
                losses = torch.log(1 + neg / pos)
                return losses.mean()  # Return average loss
            else:
                # 3D case: batched similarity computation
                sims = torch.einsum('ikm,jkm->ijk', query, clip_prompts)  # [n_query, num_prompts, B]
                batch_size = sims.size(-1)
                n_query = query.size(0)
                losses = torch.tensor(0., device=sims.device)
                for i in range(batch_size):
                    # Positive similarity for batch i
                    pos = torch.exp(sims[:, :, i][torch.arange(n_query), indices[i]])
                    # Mask for batch i
                    mask = torch.ones_like(sims[:, :, i], device=sims.device)  # [n_query, num_prompts]
                    mask[torch.arange(n_query), indices[i]] = 0
                    # Sum of negative similarities for batch i
                    neg = (torch.exp(sims[:, :, i]) * mask).sum(dim=1)  # [n_query,]
                    # Accumulate loss for batch i
                    losses += torch.log(1 + neg / pos).mean()
                return losses
        # Compute contrastive loss using the matched indices
        loss_contrastive = contrastive_loss(clip_prompts, query, indices)
        # Return matched prompts, indices, and the computed loss
        return new_prompts, indices, loss_contrastive
    
    def calculate_orthogonal_loss(self, med_tokens, target=None):
        """ Calculate orthogonalization loss for a list of tensors.
        Args:
            med_tokens: List of tensors, each with shape (Q, B, 256)
            target: Optional target tensor for computing the target similarity matrix
        Returns:
            loss_orth: Dictionary containing orthogonal losses for each tensor in the list
        """
        loss_orth = {}
        for i, tensor in enumerate(med_tokens):
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(1)
            Q, B, D = tensor.shape
            # Normalize the tensor (L2 normalization)
            normalized_tensor = F.normalize(tensor, p=2, dim=-1).transpose(0, 1)  # Shape: (B, Q, 256)
            # Compute cosine similarity matrix using batch matrix multiplication
            S = torch.bmm(normalized_tensor, normalized_tensor.transpose(1, 2))  # Shape: (B, Q, Q)
            # Create identity matrix as target (diagonals are 1, others are 0)
            if target is None:
                I = torch.eye(Q, device=S.device).unsqueeze(0).expand(B, Q, Q)  # Shape: (B, Q, Q)
            else:
                # Normalize target and compute its similarity matrix
                normalized_target = F.normalize(target, p=2, dim=-1).transpose(0, 1)  # Shape: (B, Q, 256)
                I = torch.bmm(normalized_target, normalized_target.transpose(1, 2))
                I = I[:, :Q, :Q]  # Truncate to match the size of S
            # Compute orthogonal loss using mean squared error (MSE) between S and I
            loss = F.mse_loss(S, I, reduction='mean')
            # Store the loss in the dictionary with appropriate key
            if i > 0:
                loss_orth.update({f"loss_orth_{i-1}": loss})
            else:
                loss_orth.update({f"loss_orth": loss})
            # Break loop if exceeding the specified number of layers for orthogonality
            if i > self.orth_layers:
                break
        return loss_orth