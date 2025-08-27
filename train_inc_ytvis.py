"""
ECLIPSE
Copyright (c) 2024-present NAVER Cloud Corp.
CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)
The implementation is based on fcdl94/CoMFormer and facebookresearch/Mask2Former.
"""

"""
MaskFormer Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
import warnings
warnings.filterwarnings('ignore')

import copy
import itertools
import logging
import wandb
import os
import weakref
from collections import OrderedDict
from typing import Any, Dict, List, Set
import torch
from fvcore.nn.precise_bn import get_bn_modules

# Detectron
from detectron2.modeling import build_model
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, get_detection_dataset_dicts, DatasetMapper, DatasetCatalog
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.evaluation import (
    DatasetEvaluators,
    DatasetEvaluator,
    inference_on_dataset,
    print_csv_format,
    verify_results,
    COCOEvaluator,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger
from detectron2.engine.train_loop import SimpleTrainer, AMPTrainer, TrainerBase
from detectron2.engine import hooks
from detectron2.engine.defaults import create_ddp_model, default_writers

# MaskFormer
from mask2former import (
    InstanceSegEvaluator,
    #COCOInstanceNewBaselineDatasetMapper,
    #MaskFormerInstanceDatasetMapper,
    MaskFormerSemanticDatasetMapper,
    SemanticSegmentorWithTTA,
    add_maskformer2_config,
    MaskFormerPanopticDatasetMapper,
)

from continual import add_continual_config, add_crisp_config
from continual.data import ContinualDetectron, InstanceContinualDetectron
from continual.evaluation import ContinualSemSegEvaluator, ContinualCOCOPanopticEvaluator, ContinualCOCOEvaluator
from continual.method_wrapper import build_wrapper
from continual.utils.hooks import BetterPeriodicCheckpointer, BetterEvalHook
from continual.modeling.classifier import WA_Hook
from continual.data import MaskFormerInstanceDatasetMapper, COCOInstanceNewBaselineDatasetMapper

# ytvis2019
from crisp_dataset import YTVIS2019SegmentationIncremental
from mask2former_video import (
    YTVISDatasetMapper,
    YTVISEvaluator,
    add_maskformer2_video_config,
    get_detection_dataset_dicts,
)
import tasks
from mask2former_video import (
    YTVISDatasetMapper,
    YTVISEvaluator,
    add_maskformer2_video_config,
    get_detection_dataset_dicts,
)
from copy import deepcopy
from draw_weight import draw_weight, draw_correlation, draw_cov, draw_norm
from detectron2.data.samplers import InferenceSampler
from crisp_utils import build_model_2, build_train_loader_v2, build_incremental_evaluator, build_test_loader, pca_init

from detectron2.data.build import (
    trivial_batch_collator,
)
from itertools import cycle

from torch.utils.data.distributed import DistributedSampler
import torch.nn as nn

class IncrementalTrainer(TrainerBase):
    """
    Extension of the Trainer class adapted to Continual MaskFormer.
    """

    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        super().__init__()
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        exp_id = cfg.NAME.split("_")[-1]
        # ytvis task
        self.task = cfg.CONT.TASK_NAME.split("-")[-1]
        logger.info(f"task_name={self.task}")
        self.step = cfg.CONT.TASK 
        classes = tasks.get_per_task_classes(dataset='vis', name=self.task, step=self.step)
        self.classes = classes # num_classes list of per step
        self.logger = logger
        model = self.build_model(cfg, classes=classes, logger=logger)
        self.model_old = self.build_model(cfg, old=True,logger=logger) if cfg.CONT.TASK > 0 and cfg.CONT.NUM_PROMPTS == 0 else None

        logger.info(f"current step={self.step}")
        logger.info(f"current model load from {cfg.MODEL.WEIGHTS}")

        
        self.optimizer = optimizer = self.build_optimizer(cfg, model)
        self.data_loader = data_loader = self.build_train_loader(cfg, logger=logger)

        self.model = model = create_ddp_model(model, broadcast_buffers=False, find_unused_parameters=False)
        model_wrapper = build_wrapper(cfg, model, self.model_old)
        if cfg.CONT.TASK > 0:
            self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
                model, data_loader, optimizer
            )
        else:
            self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = DetectionCheckpointer(
            model,
            cfg.OUTPUT_DIR,
            trainer=weakref.proxy(self),
        )
        if self.model_old is not None:
            self.checkpointer_old = DetectionCheckpointer(self.model_old, cfg.OUTPUT_DIR)
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self._last_eval_results = None
        self.register_hooks(self.build_hooks())
        
    def resume_or_load(self, resume=True):
        """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.

        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.
        """
        self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume)
        if self.cfg.CONT.NUM_PROMPTS > 0:
            # DDP or DP
            if isinstance(self.model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
                self.model.module.copy_prompt_embed_weights()
                self.model.module.copy_mask_embed_weights()
                self.model.module.copy_no_obj_weights()
                self.model.module.copy_prompt_trans_decoder_weights()
                if self.cfg.CRISP.INIT_PROMPTS:
                    self.init_prompt()
            else:
                self.model.copy_prompt_embed_weights()
                self.model.copy_mask_embed_weights()
                self.model.copy_no_obj_weights()
                self.model.copy_prompt_trans_decoder_weights()
                if self.cfg.CRISP.INIT_PROMPTS:
                    self.init_prompt()
        if self.model_old is not None:
            # We never want to resume it! Resume = False even when loading from checkpoint.
            # This should be strict. If you see that the model does not use some parameters, there's an error.
            self.checkpointer_old.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=False)
        if resume and self.checkpointer.has_checkpoint():
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration
            self.start_iter = self.iter + 1

    def build_hooks(self):
        """
        Taken from DefaultTrainer (detectron2.engine.defaults)

        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(BetterPeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))

        def test_and_save_results():
            self.cfg.defrost()
            x = self.cfg.DATASETS.TEST[0]
            self.cfg.DATASETS.TEST = (x.replace("train", "val"),)
            self.cfg.freeze()
            print(self.cfg.DATASETS.TEST)
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results
        def test_on_split_and_save(valid_on_cur_task_data):
            def write_dict_to_txt(dictionary, file_path):
                """
                Writes a dictionary to a text file, with each key-value pair on a separate line.
                """
                try:
                    with open(file_path, 'w', encoding='utf-8') as file:
                        for key, value in dictionary.items():
                            file.write(f"{key}: {value}\n")
                    print(f"Dictionary has been successfully written to {file_path}")
                except Exception as e:
                    print(f"Error writing to file: {e}")
            self.cfg.defrost()
            x = self.cfg.DATASETS.TEST[0]
            self.cfg.DATASETS.TEST = (x.replace("val", "train"),)
            self.cfg.freeze()
            # valid_on_cur_task_data = True
            torch.cuda.empty_cache()
            with torch.no_grad():
                self._last_eval_results = self.test_on_split(self.cfg, self.model, None, valid_on_cur_task_data)
                suffix = "cur" if valid_on_cur_task_data else 'all'
                eval_res_file = os.path.join(self.cfg.OUTPUT_DIR, f'eval_results_{suffix}.txt')
                if comm.is_main_process():
                    write_dict_to_txt(self._last_eval_results['segm'], file_path=eval_res_file)
            return self._last_eval_results
        if 'ytvis' in cfg.DATASETS.TRAIN[0]:
            from functools import partial
            test_on_split_and_save_all = partial(test_on_split_and_save, valid_on_cur_task_data=False)
            test_on_split_and_save_cur = partial(test_on_split_and_save, valid_on_cur_task_data=True)
            # Do evaluation after checkpointer, because then if it fails,
            # we can use the saved checkpoint to debug.
            # ret.append(BetterEvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))
            if "ytvis" in cfg.INPUT.DATASET_MAPPER_NAME:
                ret.append(BetterEvalHook(cfg.TEST.EVAL_PERIOD, test_on_split_and_save_all))
                ret.append(BetterEvalHook(cfg.TEST.EVAL_PERIOD, test_on_split_and_save_cur))
        
        if comm.is_main_process():
            # Here the default print/log frequency of each writer is used.
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))

        if self.cfg.CONT.WA_STEP > 0 and self.cfg.CONT.TASK > 0:
            ret.append(WA_Hook(model=self.model, step=100, distributed=True))
        return ret

    def build_writers(self):
        """
        Build a list of writers to be used using :func:`default_writers()`.
        If you'd like a different list of writers, you can overwrite it in
        your trainer.

        Returns:
            list[EventWriter]: a list of :class:`EventWriter` objects.
        """
        return default_writers(self.cfg.OUTPUT_DIR, self.max_iter)

    def train(self):
        """
        Run training.

        Returns:
            OrderedDict of results, if evaluation is enabled. Otherwise None.
        """

        super().train(self.start_iter, self.max_iter)
        if len(self.cfg.TEST.EXPECTED_RESULTS) and comm.is_main_process():
            assert hasattr(
                self, "_last_eval_results"
            ), "No evaluation results obtained during training!"
            verify_results(self.cfg, self._last_eval_results)
        self.write_results(self._last_eval_results)
        return self._last_eval_results

    def run_step(self):
        self._trainer.iter = self.iter
        # print("iter: ", self.iter)
        self._trainer.run_step()

    @classmethod
    def build_model(cls, cfg, old=False, classes=[],logger=None):

        if old:
            cfg = cfg.clone()
            cfg.defrost()
            cfg.CONT.TASK -= 1
            logger.info("<<<<<<<<<This is OLD model>>>>>>>>>>")
        model = build_model_2(cfg, classes)
        
        coco_pre = "./model_final_3c8ec9.pkl"
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            coco_pre
        )
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS
        )
        if not old:
            logger = logging.getLogger(__name__)
            logger.info("Model:\n{}".format(model))
            use_prompt_adapter = False
            if (cfg.CONT.TASK > 0 and (cfg.CONT.NUM_PROMPTS or use_prompt_adapter) > 0 and cfg.CONT.BACKBONE_FREEZE):
                logger.info("Model Freezing for Visual Prompt Tuning")
                model.freeze_for_prompt_tuning()
        else:
            model.model_old = True  # we need to set this as the old model.
            model.sem_seg_head.predictor.set_as_old_model()
            model.eval()  # and we set it to eval mode.
            for par in model.parameters():
                par.requires_grad = False
        return model

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )
        
        logger = logging.getLogger(__name__)
        unfreeze_params = []
        freeze_params = []
        
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    freeze_params.append(f"{module_name}.{module_param_name}")
                    continue
                    
                unfreeze_params.append(f"{module_name}.{module_param_name}")
                
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if "sem_seg_head.predictor.mask_embed" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.HEAD_MULTIPLIER
                # if "sem_seg_head.predictor.class_embed" in module_name:
                #     hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.HEAD_MULTIPLIER
                if (
                        "relative_position_bias_table" in module_param_name
                        or "absolute_pos_embed" in module_param_name
                ):
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                    # hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.HEAD_MULTIPLIER
                params.append({"params": [value], **hyperparams})

        if comm.is_main_process():
            print("\n============ freeze parameters ============")
            for param in freeze_params:
                print(f"   {param}")
            print("============ freeze parameters ============\n")

            print("\n============ unfreeze parameters ============")
            for param in unfreeze_params:
                print(f"   {param}")
            print("============ unfreeze parameters ============\n")
            
            num_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            storage_size = sum(
                p.numel() * p.element_size() for p in model.parameters() if p.requires_grad
            ) / (1024 ** 2)  # Convert to MB
            
            print(f"============ num_params: {num_params//1000} K" )
            print(f"============ storage_size: {storage_size} MB" )
                
        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                    cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                    and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                    and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.projects.deeplab.build_lr_scheduler`.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_train_loader(cls, cfg, logger=None):
        # Semantic segmentation dataset mapper
        if cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_semantic":  # and "voc" in cfg.DATASETS.TRAIN[0]:
            mapper = MaskFormerSemanticDatasetMapper(cfg, True)
            wrapper = ContinualDetectron
            n_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES - 1
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_instance_lsj":
            mapper = COCOInstanceNewBaselineDatasetMapper(cfg, True)
            wrapper = InstanceContinualDetectron
            n_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_instance":
            mapper = MaskFormerInstanceDatasetMapper(cfg, True)
            wrapper = InstanceContinualDetectron
            n_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_panoptic":
            mapper = MaskFormerPanopticDatasetMapper(cfg, True)
            wrapper = ContinualDetectron
            n_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES - 1  # we have bkg
        elif cfg.INPUT.DATASET_MAPPER_NAME == "ytvis2019":


            task = cfg.CONT.TASK_NAME.split("-")[-1]
            logger.info(f"task_name={task}")
            step = cfg.CONT.TASK
            cls_labels, _ = tasks.get_task_labels(dataset='vis', name=task, step=step)

            world_size = comm.get_world_size() 
            rank = comm.get_rank()
            logger.info(f"world_size={world_size}, rank={rank}")
            mapper= YTVISDatasetMapper(cfg)
            dataset_name = cfg.DATASETS.TRAIN[0]
            data_dict = get_detection_dataset_dicts(
                        dataset_name,
                        filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
                        proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
            )
            if 'ovis' in cfg.DATASETS.TRAIN[0]:
                data_dir = "./datasets/ovis/"
            elif 'ytvis_2019' in cfg.DATASETS.TRAIN[0]:
                data_dir = "./datasets/ytvis_2019/"
            elif 'ytvis_2021' in cfg.DATASETS.TRAIN[0]:
                data_dir = "./datasets/ytvis_2021/"
            logger.info(f"cur_task_labels={cls_labels}")
            train_dst_all = YTVIS2019SegmentationIncremental(cfg, data_dir, state='train_split',
                                                    mapper=mapper, dataset_dict=data_dict, labels=cls_labels)
            logger.info(f"train_dst_all_length={len(train_dst_all)}")

            train_loader = build_train_loader_v2(train_dst_all, batch_size=cfg.CRISP.BATCH_SIZE_PER_GPU, 
                                        sampler=DistributedSampler(train_dst_all, num_replicas=world_size, rank=rank, drop_last=True))

            return cycle(train_loader)

        else:
            raise NotImplementedError("At the moment, we support only segmentation")
        dataset = get_detection_dataset_dicts(
            cfg.DATASETS.TRAIN,
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
            if cfg.MODEL.KEYPOINT_ON
            else 0,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
        )
        scenario = wrapper(
            dataset,
            # Continuum related:
            initial_increment=cfg.CONT.BASE_CLS, increment=cfg.CONT.INC_CLS,
            nb_classes=n_classes,
            save_indexes=os.getenv("DETECTRON2_DATASETS", "datasets") + '/' + cfg.TASK_NAME,
            mode=cfg.CONT.MODE, class_order=cfg.CONT.ORDER,
            # Mask2Former related:
            mapper=mapper, cfg=cfg
        )
        return scenario[cfg.CONT.TASK]

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable
        """
        if not hasattr(cls, "scenario"):
            mapper = DatasetMapper(cfg, False)
            if cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_semantic":
                wrapper = ContinualDetectron
                n_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES - 1
            elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_instance_lsj":
                mapper = COCOInstanceNewBaselineDatasetMapper(cfg, False)
                wrapper = InstanceContinualDetectron
                n_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
            elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_instance":
                mapper = MaskFormerInstanceDatasetMapper(cfg, False)
                wrapper = InstanceContinualDetectron
                n_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
            elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_panoptic":
                mapper = MaskFormerPanopticDatasetMapper(cfg, False)
                wrapper = ContinualDetectron
                n_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES - 1  # we have bkg
            elif cfg.INPUT.DATASET_MAPPER_NAME == "ytvis2019":
                test_data_loader = build_test_loader(cfg, dataset_name)
                return test_data_loader
            elif cfg.INPUT.DATASET_MAPPER_NAME == "vspw":
                
                test_data_loader = build_test_loader(cfg, dataset_name)
                return test_data_loader
            else:
                raise NotImplementedError("At the moment, we support only segmentation")

            dataset = get_detection_dataset_dicts(
                dataset_name,
                filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
                proposal_files=None,
            )
            scenario = wrapper(
                dataset,
                # Continuum related:
                initial_increment=cfg.CONT.BASE_CLS, increment=cfg.CONT.INC_CLS,
                nb_classes=n_classes,
                save_indexes=os.getenv("DETECTRON2_DATASETS", "datasets") + '/' + cfg.TASK_NAME,
                mode=cfg.CONT.MODE, class_order=cfg.CONT.ORDER,
                # Mask2Former related:
                mapper=mapper, cfg=cfg, masking_value=0,
            )
            cls.scenario = scenario[cfg.CONT.TASK]
        else:
            print("Using computed scenario.")
        return cls.scenario

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each
        builtin dataset. For your own dataset, you can simply create an
        evaluator manually in your script and do not have to worry about the
        hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        os.makedirs(output_folder, exist_ok=True)
        if 'ytvis' in dataset_name or 'vspw' in dataset_name:
            return YTVISEvaluator(dataset_name, cfg, True, output_folder)
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        # semantic segmentation
        if evaluator_type in ["sem_seg"]:  # , "ade20k_panoptic_seg"]:
            evaluator_list.append(
                ContinualSemSegEvaluator(
                    cfg,
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )
        if evaluator_type == "coco":
            evaluator_list.append(ContinualCOCOEvaluator(cfg, dataset_name, output_dir=output_folder))
        if evaluator_type == "ade20k_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
            evaluator_list.append(InstanceSegEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "ade20k_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON:
            evaluator_list.append(ContinualCOCOPanopticEvaluator(cfg, dataset_name, output_folder))
            
        if evaluator_type in [
            "cityscapes_panoptic_seg",
            "mapillary_vistas_panoptic_seg",
            "coco_panoptic_seg",
        ]:
            if cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON:
                evaluator_list.append(ContinualCOCOPanopticEvaluator(cfg, dataset_name, output_folder))

        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Evaluate the given model. The given model is expected to already contain
        weights to evaluate.

        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.

        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            with torch.amp.autocast('cuda'):
                results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results
    
    @staticmethod
    def auto_scale_workers(cfg, num_workers: int):
        """
        Taken from DefaultTrainer (detectron2.engine.defaults)
        """
        old_world_size = cfg.SOLVER.REFERENCE_WORLD_SIZE
        if old_world_size == 0 or old_world_size == num_workers:
            return cfg
        cfg = cfg.clone()
        frozen = cfg.is_frozen()
        cfg.defrost()

        assert (
                cfg.SOLVER.IMS_PER_BATCH % old_world_size == 0
        ), "Invalid REFERENCE_WORLD_SIZE in config!"
        scale = num_workers / old_world_size
        bs = cfg.SOLVER.IMS_PER_BATCH = int(round(cfg.SOLVER.IMS_PER_BATCH * scale))
        lr = cfg.SOLVER.BASE_LR = cfg.SOLVER.BASE_LR * scale
        max_iter = cfg.SOLVER.MAX_ITER = int(round(cfg.SOLVER.MAX_ITER / scale))
        warmup_iter = cfg.SOLVER.WARMUP_ITERS = int(round(cfg.SOLVER.WARMUP_ITERS / scale))
        cfg.SOLVER.STEPS = tuple(int(round(s / scale)) for s in cfg.SOLVER.STEPS)
        cfg.TEST.EVAL_PERIOD = int(round(cfg.TEST.EVAL_PERIOD / scale))
        cfg.SOLVER.CHECKPOINT_PERIOD = int(round(cfg.SOLVER.CHECKPOINT_PERIOD / scale))
        cfg.SOLVER.REFERENCE_WORLD_SIZE = num_workers  # maintain invariant
        logger = logging.getLogger(__name__)
        logger.info(
            f"Auto-scaling the config to batch_size={bs}, learning_rate={lr}, "
            f"max_iter={max_iter}, warmup={warmup_iter}."
        )

        if frozen:
            cfg.freeze()
        return cfg

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA.
        logger.info("Running inference with test-time augmentation ...")
        model = SemanticSegmentorWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

    def state_dict(self):
        ret = super().state_dict()
        ret['trainer'] = self._trainer.state_dict()

        return ret

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self._trainer.load_state_dict(state_dict['trainer'])

    def write_results(self, results):
        name = self.cfg.NAME
        path = f"results/{self.cfg.TASK_NAME}.csv"
        path_acc = f"results/{self.cfg.TASK_NAME}_acc.csv"
        if "sem_seg" in results:
            res = results['sem_seg']
            cls_iou = []
            cls_acc = []
            for k in res:
                if k.startswith("IoU-"):
                    cls_iou.append(res[k])
                if k.startswith("ACC-"):
                    cls_acc.append(res[k])

            with open(path, "a") as out:
                out.write(f"{name},{self.cfg.CONT.TASK},{res['mIoU_base']},{res['mIoU_novel']},{res['mIoU']},")
                out.write(",".join([str(i) for i in cls_iou]))
                out.write("\n")

            with open(path_acc, "a") as out:
                out.write(f"{name},{self.cfg.CONT.TASK},{res['mACC']},{res['pACC']},-,")
                out.write(",".join([str(i) for i in cls_acc]))
                out.write("\n")
        if 'panoptic_seg' in results:
            res = results['panoptic_seg']
            cls_pq = OrderedDict()
            cls_rq = OrderedDict()
            cls_sq = OrderedDict()
            for k in res:
                if k.startswith("PQ_c"):
                    cls_pq[int(k[4:])] = res[k]
                if k.startswith("RQ_c"):
                    cls_rq[int(k[4:])] = res[k]
                if k.startswith("SQ_c"):
                    cls_sq[int(k[4:])] = res[k]
            with open(path, "a") as out:
                out.write(f"{name},{self.cfg.CONT.TASK},{res['PQ']},{res['RQ']},{res['SQ']},")
                out.write(",".join([str(i) for i in cls_pq.values()]))
                out.write(f",")
                out.write(",".join([str(i) for i in cls_rq.values()]))
                out.write(f",")
                out.write(",".join([str(i) for i in cls_sq.values()]))
                out.write("\n")
        if 'segm' in results:
            res = results['segm']
            path = f"results/{self.cfg.TASK_NAME}.csv"
            class_ap = []
            for k in res:
                if k.startswith("AP-"):
                    class_ap.append(res[k])
            with open(path, "a") as out: # "AP", "AP50", "AP75", "APs", "APm", "APl"
                out.write(f"{name},{self.cfg.CONT.TASK},{res['AP']},{res['AP50']},{res['AP75']},")
                out.write(",".join([str(i) for i in class_ap]))
                out.write("\n")

    def init_prompt(self):
        if isinstance(self.model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
            origin_query = deepcopy(self.model.module.sem_seg_head.predictor.query_feat.weight.data)
            prompt_feat = self.model.module.sem_seg_head.predictor.prompt_feat # module list
            prompt_feat_recorder = [origin_query]
            for i, t_p in enumerate(prompt_feat):
                t_p = t_p.weight.data
                prompt_feat_recorder.append(t_p)
            num_prompts = prompt_feat[-1].weight.data.shape[0]
            prompt_feat_all = torch.cat(prompt_feat_recorder[:-1])
            n_com = prompt_feat_all.shape[0]
            new_prompt = pca_init(prompt_feat_all, n_components=n_com, mode='max', num_components=num_prompts, scale=True)
            self.model.module.sem_seg_head.predictor.prompt_feat[-1].weight.data = torch.tensor(-new_prompt, dtype=torch.float32).to(self.model.module.device)
        else:
            origin_query = deepcopy(self.model.sem_seg_head.predictor.query_feat.weight.data)
            prompt_feat = self.model.sem_seg_head.predictor.prompt_feat # module list
            prompt_feat_recorder = [origin_query]
            for i, t_p in enumerate(prompt_feat):
                t_p = t_p.weight.data
                prompt_feat_recorder.append(t_p)
            num_prompts = prompt_feat[-1].weight.data.shape[0]
            prompt_feat_all = torch.cat(prompt_feat_recorder[:-1])
            n_com = prompt_feat_all.shape[0]
            new_prompt = pca_init(prompt_feat_all, n_components=n_com, mode='max', num_components=num_prompts, scale=True)
            self.model.sem_seg_head.predictor.prompt_feat[-1].weight.data = torch.tensor(-new_prompt, dtype=torch.float32).to(self.model.device)
    @classmethod
    def test_on_split(cls, cfg, model, evaluators=None, valid_on_cur_task_data=False):
            """
            Evaluate the given model. The given model is expected to already contain
            weights to evaluate.
            Args:
                cfg (CfgNode):
                model (nn.Module):
                evaluators (list[DatasetEvaluator] or None): if None, will call
                    :meth:`build_evaluator`. Otherwise, must have the same length as
                    ``cfg.DATASETS.TEST``.
            Returns:
                dict: a dict of result metrics
            """
            
            logger = logging.getLogger(__name__)
            if isinstance(evaluators, DatasetEvaluator):
                evaluators = [evaluators]
            if evaluators is not None:
                assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                    len(cfg.DATASETS.TEST), len(evaluators)
                )

            results = OrderedDict()
            for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
                mapper= YTVISDatasetMapper(cfg, is_train=False)
                # dataset_name = cfg.DATASETS.TRAIN[0]
                data_dict = get_detection_dataset_dicts(
                            dataset_name,
                            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
                            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
                )
                task = cfg.CONT.TASK_NAME.split("-")[-1]
                step = cfg.CONT.TASK
                incremental = valid_on_cur_task_data
                cls_labels, cls_labels_old = tasks.get_task_labels(dataset='vis', name=task, step=step)
                if 'ovis' in cfg.DATASETS.TEST[0]:
                    data_dir = "./datasets/ovis/"
                elif 'ytvis_2019' in cfg.DATASETS.TEST[0]:
                    data_dir = "./datasets/ytvis_2019/"
                elif 'ytvis_2021' in cfg.DATASETS.TEST[0]:
                    data_dir = "./datasets/ytvis_2021/"
                dataset = YTVIS2019SegmentationIncremental(cfg, data_dir, state='test_split',
                                                        mapper=mapper, dataset_dict=data_dict, labels=cls_labels, 
                                                        train=False, valid_on_cur_task_data=valid_on_cur_task_data)
                currentTask_vidIDs = list(map(lambda x: x + 1, dataset.dataset.subset_ids))
                currentTask_catIDs = list(map(lambda x: x + 1, cls_labels))
                print(len(dataset))
                sampler = InferenceSampler(len(dataset))
                # Always use 1 image per worker during inference since this is the
                # standard when reporting inference time in papers.
                batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, 1, drop_last=False)
                data_loader = torch.utils.data.DataLoader(
                dataset,
                num_workers=4,
                batch_sampler=batch_sampler,
                collate_fn=trivial_batch_collator,
                )
                # When evaluators are passed in as arguments,
                # implicitly assume that evaluators can be created before data_loader.
                if evaluators is not None:
                    evaluator = evaluators[idx]
                else:
                    try:
                        evaluator = build_incremental_evaluator(cfg, dataset_name)
                        evaluator.incremental = incremental
                        evaluator.currentTask_vidIDs = currentTask_vidIDs
                        evaluator.currentTask_catIDs = currentTask_catIDs
                    except NotImplementedError:
                        logger.warn(
                            "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                            "or implement its `build_evaluator` method."
                        )
                        results[dataset_name] = {}
                        continue
                with torch.amp.autocast('cuda'):
                    results_i = inference_on_dataset(model, data_loader, evaluator)
                results[dataset_name] = results_i
                if comm.is_main_process():
                    assert isinstance(
                        results_i, dict
                    ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                        results_i
                    )
                    logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                    print_csv_format(results_i)

            if len(results) == 1:
                results = list(results.values())[0]
            return results
    
def write_settings(cfg, save_dir='./'):

    import csv

    setting_dict={
        'weights':cfg.CONT.WEIGHTS,
        "base_queries":cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
        "num_prompts":cfg.CONT.NUM_PROMPTS, #

        'iter':cfg.SOLVER.MAX_ITER, #
        'steps': cfg.SOLVER.STEPS, # 
        'base_lr':cfg.SOLVER.BASE_LR, #

        'dice_weight':cfg.MODEL.MASK_FORMER.DICE_WEIGHT,
        'mask_weight':cfg.MODEL.MASK_FORMER.MASK_WEIGHT,
        'class_weight':cfg.MODEL.MASK_FORMER.CLASS_WEIGHT,

        'backbone_freeze':cfg.CONT.BACKBONE_FREEZE,
        'trans_decoder_freeze':cfg.CONT.TRANS_DECODER_FREEZE,
        'pixel_decoder_freeze':cfg.CONT.PIXEL_DECODER_FREEZE,
        'cls_head_freeze':cfg.CONT.CLS_HEAD_FREEZE,
        'mask_head_freeze':cfg.CONT.MASK_HEAD_FREEZE,
        'query_embed_freeze':cfg.CONT.QUERY_EMBED_FREEZE,

        'prompt_deep':cfg.CONT.PROMPT_DEEP,
        'prompt_mask_mlp':cfg.CONT.PROMPT_MASK_MLP,
        'prompt_no_obj_mlp':cfg.CONT.PROMPT_NO_OBJ_MLP,

        'temp':None,
        'deltas':cfg.CONT.LOGIT_MANI_DELTAS,

        'deep_cls':cfg.CONT.DEEP_CLS,

        'use_clip_prompter': cfg.CRISP.USE_CLIP_PROMPTER,
        'without_text_encoder': cfg.CRISP.WITHOUT_TEXT_ENCODER,
        'init_prompts': cfg.CRISP.INIT_PROMPTS,
        'contrastive_loss': cfg.CRISP.USE_CONTRASTIVE_LOSS, 
        'contrastive_weight': cfg.CRISP.CONTRASTIVE_WEIGHT,
        'infer_with_clip_prompter': cfg.CRISP.INFER_WITH_CLIP_PROMPTER,
        'orth_loss': cfg.CRISP.USE_ORTH_LOSS,
        'orth_weight': cfg.CRISP.ORTH_WEIGHT,
        'orth_layers': cfg.CRISP.ORTH_LAYERS,

    }
    save_path = os.path.join(save_dir, 'settings.csv')
    with open(save_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Key', 'Value'])  # 写入表头
        for key, value in setting_dict.items():
            if isinstance(value, list):
                value_str = ', '.join(map(str, value))
            else:
                value_str = str(value)
            writer.writerow([key, value_str])
    print(f"Setting in {save_path}")

def visual_weight(model, model_path, logger):

    model.eval()
    model.is_valid = False
    model_base_dir = os.path.dirname(model_path)
    class_embed = model.sem_seg_head.predictor.class_embed.cls
    
    weight_recorder = []
    bias_recorder = []
    grad_recorder = []
    for i, mlp in enumerate(class_embed):
        # per step
        weight = mlp.layers[-1].weight.data.cpu().detach()
        bias = mlp.layers[-1].bias.data.cpu().detach()
        weight_recorder.append(weight)
        bias_recorder.append(bias)
        grad_recorder.append(bias)
        save_dir_weight = os.path.join(model_base_dir,"pictures/weight")
        if i==0:
            draw_weight(weight, save_dir_weight, cur_epoch=f"bg")
        else:
            draw_weight(weight, save_dir_weight, cur_epoch=f"step_{i-1}")
    cat_weight = torch.cat(weight_recorder, dim=0)
    draw_cov(cat_weight, os.path.join(model_base_dir, "pictures/cov"), "all")
    draw_correlation(cat_weight, os.path.join(model_base_dir, "pictures/correlation"), "all")
    draw_norm(cat_weight, os.path.join(model_base_dir, "pictures/norm"), "all")
    # prompt
    prompt_feat = model.sem_seg_head.predictor.prompt_feat # module list
    origin_query = model.sem_seg_head.predictor.query_feat.weight.data 
    save_dir = os.path.join(model_base_dir,"pictures/prompt_feat")
    draw_weight(origin_query, save_dir, f"step0")
    prompt_feat_recorder = [origin_query]
    for i, t_p in enumerate(prompt_feat):
        t_p = t_p.weight.data
        prompt_feat_recorder.append(t_p)
        draw_weight(t_p, save_dir, f"step{i+1}")
    prompt_feat_all = torch.cat(prompt_feat_recorder)
    draw_weight(prompt_feat_all, save_dir, f"all")

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.NAME = "Exp"

    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_continual_config(cfg)
    add_crisp_config(cfg)
    add_maskformer2_video_config(cfg)

    cfg.set_new_allowed(True)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.defrost()
    cfg.INPUT.DATASET_MAPPER_NAME = "ytvis2019"
    cfg.CRISP.BATCH_SIZE_PER_GPU = int(cfg.SOLVER.IMS_PER_BATCH / args.num_gpus)
    cfg.freeze()

    if cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON:
        suffix = "-pan"
    elif cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
        suffix = "-vis"
    else:
        suffix = ""

    if cfg.CONT.MODE == 'overlap':
        cfg.defrost()
        if 'ytvis' in cfg.DATASETS.TRAIN[0]:
            cfg.TASK_NAME = f"{cfg.DATASETS.TRAIN[0][:10]}{suffix}_{cfg.CONT.BASE_CLS}-{cfg.CONT.INC_CLS}-ov"
        elif 'ovis' in cfg.DATASETS.TRAIN[0]:
            cfg.TASK_NAME = f"{cfg.DATASETS.TRAIN[0][:4]}{suffix}_{cfg.CONT.BASE_CLS}-{cfg.CONT.INC_CLS}-ov"
    elif cfg.CONT.MODE == "disjoint":
        cfg.TASK_NAME = f"{cfg.DATASETS.TRAIN[0][:3]}{suffix}_{cfg.CONT.BASE_CLS}-{cfg.CONT.INC_CLS}-dis"
    else:
        cfg.TASK_NAME = f"{cfg.DATASETS.TRAIN[0][:3]}{suffix}_{cfg.CONT.BASE_CLS}-{cfg.CONT.INC_CLS}-seq"

    if cfg.CONT.ORDER_NAME is not None:
        cfg.TASK_NAME += "-" + cfg.CONT.ORDER_NAME

    cfg.OUTPUT_ROOT = cfg.OUTPUT_DIR
    cfg.OUTPUT_DIR = cfg.OUTPUT_DIR + "/" + cfg.TASK_NAME + "/" + cfg.NAME + "/step" + str(cfg.CONT.TASK)

    cfg.freeze()
    default_setup(cfg, args)

    # Setup logger for "mask_former" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former")
    if comm.get_rank() == 0 and cfg.WANDB:
        wandb.init(project="ECLIPSE", entity="your_entity",
                   name=f"{cfg.NAME}_step_{cfg.CONT.TASK}",
                   config=cfg, sync_tensorboard=True, group="PVT_"+cfg.TASK_NAME, settings=wandb.Settings(start_method="fork"))

    return cfg


def main(args):
    cfg = setup(args)
    if hasattr(cfg, 'CONT') and cfg.CONT.TASK > 0:
        cfg.defrost()
        if cfg.CONT.WEIGHTS is None:  # load from last step
            cfg.MODEL.WEIGHTS = cfg.OUTPUT_ROOT + "/" + cfg.TASK_NAME + "/" + cfg.NAME + f"/step{cfg.CONT.TASK - 1}/model_final.pth"
        else:  # load from cfg
            cfg.MODEL.WEIGHTS = cfg.CONT.WEIGHTS

        cfg.freeze()
    write_settings(cfg, save_dir=os.path.dirname(cfg.OUTPUT_DIR))
    eval_only = False
    if eval_only:
        model = IncrementalTrainer.build_model(cfg)

        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )

        res = IncrementalTrainer.test_on_split(cfg, model, None, valid_on_cur_task_data=False)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = IncrementalTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    ret = trainer.train()
    model_final = IncrementalTrainer.build_model(cfg, classes=trainer.classes, logger=trainer.logger)
    model_final_path = cfg.OUTPUT_DIR+f"/model_final.pth"
    DetectionCheckpointer(model_final, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            model_final_path, resume=args.resume
        )
    visual_weight(model_final, model_final_path, trainer.logger)

    if comm.get_rank() == 0:
        wandb.finish()
    return ret


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"# 4, 5, 6, 7
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

