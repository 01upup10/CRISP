from types import SimpleNamespace
import torch
import detectron2.utils.comm as comm
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger
from detectron2.utils.logger import _log_api_usage
from detectron2.utils.registry import Registry
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from mask2former_video import (
    YTVISDatasetMapper,
    YTVISEvaluator,
    VSPWDatasetMapper,
    IncrementalYTVISEvaluator,
    add_maskformer2_video_config,
    build_detection_train_loader,
    build_detection_test_loader,
    get_detection_dataset_dicts,
)
from mask2former import add_maskformer2_config

from typing import Any, Dict, List, Set
import copy
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.evaluation import (
    DatasetEvaluator,
    inference_on_dataset,
    print_csv_format,
    verify_results,
)
from fvcore.common.param_scheduler import (
    CosineParamScheduler,
    MultiStepParamScheduler,
    StepWithFixedGammaParamScheduler,
)
from detectron2.solver.lr_scheduler import LRMultiplier, LRScheduler, WarmupParamScheduler
import itertools

import tasks
from torchvision.transforms.functional import normalize
import torch.nn as nn
import numpy as np
from sklearn.decomposition import PCA
import logging
import os

from collections import OrderedDict
import pickle
from torch.utils.data import random_split

from continual import add_continual_config
from mask2former_video.video_maskformer_model import VideoMaskFormer
from torch.utils.data import Dataset, DataLoader


def build_model_2(cfg, classes=None):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    
    meta_arch = cfg.MODEL.META_ARCHITECTURE
    
    model = VideoMaskFormer(cfg, classes=classes)
    model.to(torch.device(cfg.MODEL.DEVICE))
    _log_api_usage("modeling.meta_arch." + meta_arch)
    return model

def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch

def build_train_loader_v2(dataset, batch_size, sampler, ):
    
    train_loader = DataLoader(dataset, 
                              batch_size=batch_size, 
                              collate_fn=trivial_batch_collator, 
                              sampler=sampler)
    return train_loader

def build_test_loader(cfg, dataset_name):
        dataset_name = cfg.DATASETS.TEST[0]
        if 'vspw' in dataset_name:
            mapper = VSPWDatasetMapper(cfg, is_train=False)
        else:
            mapper = YTVISDatasetMapper(cfg, is_train=False)
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

def build_evaluator(cfg, dataset_name='ytvis_2019_val', output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
            os.makedirs(output_folder, exist_ok=True)

        return YTVISEvaluator(dataset_name, cfg, True, output_folder)

def build_incremental_evaluator(cfg, dataset_name='ytvis_2019_val', output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference_incremental")
            os.makedirs(output_folder, exist_ok=True)

        return IncrementalYTVISEvaluator(dataset_name, cfg, True, output_folder)


def pca_init(matrix, n_components=31, mode=None, num_components=5, scale=False, discount=1, show=False):
    '''
    input: 
        matrix: shape=[old_classes, 256]
        n_components = old_classes
        mode: 'max','min','mean','random',None
        scale:bool value ,if align norm
    output:
        new_rows: shape=[new_classes, 256]
        num_components = new_classes 
    '''
    
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.cpu()
    pca = PCA(n_components=n_components)
    pca.fit(matrix)
    
    norms = np.linalg.norm(pca.components_.T, axis=0)
    components = pca.components_
    if mode == 'max':
        # Select the x columns with the longest norm
        selected_indices = np.argsort(-norms)[:num_components]
    elif mode == 'min':
        # Select the 5 columns with the shortest norm
        selected_indices = np.argsort(norms)[:num_components]
    elif mode == 'mean':
        # Select the 5 columns with the closest norm to the average value
        mean_norm = np.mean(norms)
        selected_indices = np.argsort(np.abs(norms - mean_norm))[:num_components]
    elif mode == 'random':
        # Randomly select 5 columns
        selected_indices = np.random.choice(np.arange(components.shape[1]), num_components, replace=False)
    else:
        # Select the first 5 columns
        selected_indices = range(num_components)
    
    new_rows = components[selected_indices]
    assert new_rows.shape[-1]==256, "new_rows.shape[-1]!=256"
    if scale:
        # If scale==true, align the weight norm
        nr_norms = np.linalg.norm(pca.components_.T, axis=0)
        old_model_norms = np.linalg.norm(matrix.T, axis=0)
        scale_factor = np.mean(old_model_norms) / np.mean(nr_norms)
        # discount = 1
        new_rows = scale_factor * new_rows * discount
    return new_rows



