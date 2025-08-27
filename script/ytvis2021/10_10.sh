#!/bin/bash

# export DETECTRON2_DATASETS=YOUR_DATA_ROOT
ngpus=4

cfg_file=configs/youtubevis_2021/video_maskformer2_R50_bs16_8ep.yaml
base=results/exp
step_args="CONT.BASE_CLS 10 CONT.INC_CLS 10 CONT.MODE overlap SEED 42"
task=ytvis
task_name=ytvis-10_10 # continual setting
name=MxF
meth_args="MODEL.MASK_FORMER.TEST.MASK_BG False MODEL.MASK_FORMER.PER_PIXEL False MODEL.MASK_FORMER.FOCAL True"

base_queries=100
dice_weight=5.0
mask_weight=5.0
class_weight=2.0

base_lr=0.0001
iter=750
steps=(500,)

soft_mask=False # mask softmax (True) or sigmoid (False)
soft_cls=False  # classifier softmax (True) or sigmoid( False)

num_prompts=0
deep_cls=True

weight_args="MODEL.MASK_FORMER.NUM_OBJECT_QUERIES ${base_queries} MODEL.MASK_FORMER.DICE_WEIGHT ${dice_weight} MODEL.MASK_FORMER.MASK_WEIGHT ${mask_weight} MODEL.MASK_FORMER.CLASS_WEIGHT ${class_weight} MODEL.MASK_FORMER.SOFTMASK ${soft_mask} CONT.SOFTCLS ${soft_cls} CONT.NUM_PROMPTS ${num_prompts} CONT.DEEP_CLS ${deep_cls}"

exp_name="your_base_exp_name"

comm_args="OUTPUT_DIR ${base} ${meth_args} ${step_args} ${weight_args}"
inc_args="CONT.TASK 0 SOLVER.BASE_LR ${base_lr} TEST.EVAL_PERIOD ${iter} SOLVER.CHECKPOINT_PERIOD 500000 SOLVER.MAX_ITER ${iter} SOLVER.STEPS ${steps} CONT.TASK_NAME ${task_name}"


## Train base classes
## You can skip this process if you have a step0-checkpoint.
# python train_inc_ytvis.py --num-gpus ${ngpus} --config-file ${cfg_file} ${comm_args} ${inc_args} NAME ${exp_name} WANDB False

# --------------------------------------

base_queries=100
num_prompts=10

iter=3000
steps=(2000,)
base_lr=0.0001 

dice_weight=5.0
mask_weight=5.0
class_weight=2.0

backbone_freeze=True
trans_decoder_freeze=True
pixel_decoder_freeze=True
cls_head_freeze=True
mask_head_freeze=True
query_embed_freeze=True

prompt_deep=True
prompt_mask_mlp=True
prompt_no_obj_mlp=False

temp=None
deltas=[0.5,0.6,0.6,0.6,0.6,0.6]

deep_cls=True

# <<<crisp>>>
save=True
save_iter_step=200
linear=False
use_clip_prompter=True
init_prompts=True
contrastive_loss=True
orth_loss=True
contrastive_weight=3.
orth_weight=3.

crisp_args="CRISP.USE_CLIP_PROMPTER ${use_clip_prompter} CRISP.INIT_PROMPTS ${init_prompts} CRISP.USE_CONTRASTIVE_LOSS ${contrastive_loss} CRISP.USE_ORTH_LOSS ${orth_loss} CRISP.CONTRASTIVE_WEIGHT ${contrastive_weight} CRISP.ORTH_WEIGHT ${orth_weight} CRISP.SAVE ${save} CRISP.SAVE_ITER_STEP ${save_iter_step}"

soft_mask=False
weight_args="MODEL.MASK_FORMER.NUM_OBJECT_QUERIES ${base_queries} MODEL.MASK_FORMER.DICE_WEIGHT ${dice_weight} MODEL.MASK_FORMER.MASK_WEIGHT ${mask_weight} MODEL.MASK_FORMER.CLASS_WEIGHT ${class_weight} MODEL.MASK_FORMER.SOFTMASK ${soft_mask} CONT.SOFTCLS ${soft_cls} CONT.NUM_PROMPTS ${num_prompts}"
comm_args="OUTPUT_DIR ${base} ${meth_args} ${step_args} ${weight_args}"

cont_weight="${base}/ytvis_2019-vis_10-10-ov/${exp_name}/step0/model_final.pth"
inc_args="CONT.TASK 1 SOLVER.MAX_ITER ${iter} SOLVER.BASE_LR ${base_lr} SOLVER.STEPS ${steps} CONT.TASK_NAME ${task_name} TEST.EVAL_PERIOD ${iter} SOLVER.CHECKPOINT_PERIOD 500000 CONT.WEIGHTS ${cont_weight}" # results/ade_ps/ytv-ins_20-5-ov/ytvis_2019_base20_0213/step0/model_final.pth" # results/ytvis_2019_base20_0123/step0/model_final.pth"

vpt_args="CONT.BACKBONE_FREEZE ${backbone_freeze} CONT.CLS_HEAD_FREEZE ${cls_head_freeze} CONT.MASK_HEAD_FREEZE ${mask_head_freeze} CONT.PIXEL_DECODER_FREEZE ${pixel_decoder_freeze} CONT.QUERY_EMBED_FREEZE ${query_embed_freeze} CONT.TRANS_DECODER_FREEZE ${trans_decoder_freeze} CONT.PROMPT_MASK_MLP ${prompt_mask_mlp} CONT.PROMPT_NO_OBJ_MLP ${prompt_no_obj_mlp} CONT.PROMPT_DEEP ${prompt_deep} CONT.DEEP_CLS ${deep_cls} TEST.DETECTIONS_PER_IMAGE ${temp} CONT.LOGIT_MANI_DELTAS ${deltas}"

exp_name_inc="your_continual_part_exp_name" # e.g. XXXX_ip_prompts_contrastive_orth-9_ 'ip' is PCA-init 'prompts' is ARSP 'contrastive' is ISC 'orth' is IC
python train_inc_ytvis.py --num-gpus ${ngpus} --config-file ${cfg_file} ${comm_args} ${inc_args} ${cont_args} ${dist_args} ${vpt_args} NAME ${exp_name_inc} WANDB False ${crisp_args}
for t in 2 3; do
    inc_args="CONT.TASK ${t} SOLVER.MAX_ITER ${iter} SOLVER.STEPS ${steps} SOLVER.BASE_LR ${base_lr} TEST.EVAL_PERIOD ${iter} SOLVER.CHECKPOINT_PERIOD 500000 CONT.TASK_NAME ${task_name}"

    python train_inc_ytvis.py --num-gpus ${ngpus} --config-file ${cfg_file} ${comm_args} ${inc_args} ${cont_args} ${dist_args} ${vpt_args} NAME ${exp_name_inc} WANDB False ${crisp_args}
done