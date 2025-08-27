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

# from trainIncraminal import IncraminalTrainer
# from trainIncraminal_v2 import IncraminalTrainer_v2
# from trainIncraminal_v3 import IncraminalTrainer_v3
# import transform
import tasks
from torchvision.transforms.functional import normalize
import torch.nn as nn
import numpy as np
import logging
import os

from collections import OrderedDict
import pickle
from torch.utils.data import random_split

from continual import add_continual_config
# args = SimpleNamespace(config_file='./configs/youtubevis_2019/video_maskformer2_R50_bs16_8ep.yaml',
# dist_url='tcp://127.0.0.1:50152',
# eval_only=False,
# machine_rank=0,
# num_gpus=2,
# num_machines=1,
# opts=[],
# resume=False)

def setup(args, output_dir=None, random_seed=None):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.NAME = "Exp"
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_continual_config(cfg)
    add_maskformer2_video_config(cfg)
    cfg.set_new_allowed(True)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if random_seed != None:
        # cfg.defrost()
        cfg.SEED = random_seed
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(name="mask2former")
    if output_dir==None:
        setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former_video")
    else:
        setup_logger(output=output_dir, distributed_rank=comm.get_rank(), name="mask2former_video")
    return cfg

# cfg = setup(args)
# META_ARCH_REGISTRY = Registry("META_ARCH")  # noqa F401 isort:skip
# META_ARCH_REGISTRY.__doc__ = """
# Registry for meta-architectures, i.e. the whole model.

# The registered object will be called with `obj(cfg)`
# and expected to return a `nn.Module` object.
# """

from mask2former_video.video_maskformer_model import VideoMaskFormer
# from mask2former_video.video_maskformer_model_v2 import VideoMaskFormer
# print('查看所有已注册的模型名称',META_ARCH_REGISTRY._name)  # 查看所有已注册的模型名称
def build_lr_scheduler(cfg, optimizer: torch.optim.Optimizer) -> LRScheduler:
    """
    Build a LR scheduler from config.
    """
    name = cfg.SOLVER.LR_SCHEDULER_NAME

    if name == "WarmupMultiStepLR":
        steps = [x for x in cfg.SOLVER.STEPS if x <= cfg.SOLVER.MAX_ITER]
        if len(steps) != len(cfg.SOLVER.STEPS):
            logger = logging.getLogger(__name__)
            logger.warning(
                "SOLVER.STEPS contains values larger than SOLVER.MAX_ITER. "
                "These values will be ignored."
            )
        sched = MultiStepParamScheduler(
            values=[cfg.SOLVER.GAMMA**k for k in range(len(steps) + 1)],
            milestones=steps,
            num_updates=cfg.SOLVER.MAX_ITER,
        )
    elif name == "WarmupCosineLR":
        end_value = cfg.SOLVER.BASE_LR_END / cfg.SOLVER.BASE_LR
        assert end_value >= 0.0 and end_value <= 1.0, end_value
        sched = CosineParamScheduler(1, end_value)
    elif name == "WarmupStepWithFixedGammaLR":
        sched = StepWithFixedGammaParamScheduler(
            base_value=1.0,
            gamma=cfg.SOLVER.GAMMA,
            num_decays=cfg.SOLVER.NUM_DECAYS,
            num_updates=cfg.SOLVER.MAX_ITER,
        )
    else:
        raise ValueError("Unknown LR scheduler: {}".format(name))

    sched = WarmupParamScheduler(
        sched,
        cfg.SOLVER.WARMUP_FACTOR,
        min(cfg.SOLVER.WARMUP_ITERS / cfg.SOLVER.MAX_ITER, 1.0),
        cfg.SOLVER.WARMUP_METHOD,
        cfg.SOLVER.RESCALE_INTERVAL,
    )
    return LRMultiplier(optimizer, multiplier=sched, max_iter=cfg.SOLVER.MAX_ITER)

def build_model_2(cfg, classes=None, version=None, use_nest=False):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    
    meta_arch = cfg.MODEL.META_ARCHITECTURE
    # model = META_ARCH_REGISTRY.get(meta_arch)(cfg)
    model = VideoMaskFormer(cfg, classes=classes, use_nest=use_nest)
    model.to(torch.device(cfg.MODEL.DEVICE))
    _log_api_usage("modeling.meta_arch." + meta_arch)
    return model

from torch.utils.data import Dataset, DataLoader

def build_trainset(cfg):
    mapper = YTVISDatasetMapper(cfg, is_train=True)
    dataset_name = cfg.DATASETS.TRAIN[0]
    dataset_dict = get_detection_dataset_dicts(
                dataset_name,
                filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
                proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
            )
    return mapper, dataset_dict

def bulid_testset(cfg):
    dataset_name = cfg.DATASETS.TEST[0]
    mapper = YTVISDatasetMapper(cfg, is_train=False)
    return mapper
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
def test(cfg, model, evaluators=None):
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
        from torch.cuda.amp import autocast
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = build_test_loader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            with autocast():
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

def build_optimizer(cfg, model):
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

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})
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

def save_ckpt(path, model, trainer, optimizer, scheduler, epoch, best_score, save_as_pkl=True):
    """ save current model
    """
    
    if save_as_pkl:
        with open(path, 'wb') as f:
            pickle.dump(model, f)
    else:
        state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
            "trainer_state": trainer.state_dict()
        }
        torch.save(state, path)

from torch.optim.lr_scheduler import _LRScheduler, StepLR

class PolyLR(_LRScheduler):
    def __init__(self, optimizer, max_iters, power=0.9, last_epoch=-1):
        self.power = power
        self.max_iters = max_iters
        super(PolyLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        return [ base_lr * ( 1 - self.last_epoch/self.max_iters )**self.power
                for base_lr in self.base_lrs]
    
class DictToAccess:
    def __init__(self, dictionary):
        self.dictionary = dictionary

    def __getattr__(self, key):
        if key in self.dictionary:
            return self.dictionary[key]
        else:
            raise AttributeError(f"'DictToAccess' object has no attribute '{key}'")

def denormalize(tensor, mean, std):
    mean = np.array(mean)
    std = np.array(std)

    _mean = -mean/std
    _std = 1/std
    return normalize(tensor, _mean, _std)


class Denormalize(object):
    def __init__(self, mean, std):
        mean = np.array(mean)
        std = np.array(std)
        self._mean = -mean/std
        self._std = 1/std

    def __call__(self, tensor):
        if isinstance(tensor, np.ndarray):
            return (tensor - self._mean.reshape(-1,1,1)) / self._std.reshape(-1,1,1)
        return normalize(tensor, self._mean, self._std)


def fix_bn(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            m.weight.requires_grad = False
            m.bias.requires_grad = False


def color_map(dataset):
    if dataset=='vis':
        return voc_cmap()
    elif dataset=='cityscapes':
        return cityscapes_cmap()
    # elif dataset=='ade':
    #     return ade_cmap()


def cityscapes_cmap():
    return np.array([(128, 64,128), (244, 35,232), ( 70, 70, 70), (102,102,156), (190,153,153), (153,153,153), (250,170, 30), 
                         (220,220,  0), (107,142, 35), (152,251,152), ( 70,130,180), (220, 20, 60), (255,  0,  0), (  0,  0,142), 
                         (  0,  0, 70), (  0, 60,100), (  0, 80,100), (  0,  0,230), (119, 11, 32), (  0,  0,  0)], 
                         dtype=np.uint8)
        
def voc_cmap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap


class Label2Color(object):
    def __init__(self, cmap):
        self.cmap = cmap

    def __call__(self, lbls):
        return self.cmap[lbls]


def convert_bn2gn(module):
    mod = module
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        num_features = module.num_features
        num_groups = num_features//16
        mod = nn.GroupNorm(num_groups=num_groups, num_channels=num_features)
    for name, child in module.named_children():
        mod.add_module(name, convert_bn2gn(child))
    del module
    return mod

import numpy as np

def compute_metrics(scores, mask, gt_mask_tensor, pred_class, true_class):
    """
    计算Overall Acc, Mean IoU, Class IoU, Class Acc。

    参数:
    mask -- 预测的mask, 尺寸为B*H*W
    gt_mask -- 真实的mask, 尺寸为B*H*W
    pred_class -- 预测的类别
    true_class -- 真实的类别

    返回:
    overall_acc -- 整体准确度
    mean_iou -- 平均交并比
    class_iou -- 类别交并比
    class_acc -- 类别准确度
    """

    max_score_idx = top_n_indices(scores, len(true_class[0]))
    mask = [mask[i] for i in max_score_idx ]
    mask = torch.stack(mask)

    gt_mask_tensor = torch.nn.functional.interpolate(
                gt_mask_tensor, size=(mask.shape[-2], mask.shape[-1]), mode="bilinear", align_corners=False
            )
    gt_mask = gt_mask_tensor > 0.
    # new_mask = torch.where(mask > 0.5, 1 ,0)
    
    pred_class = [pred_class[i] for i in max_score_idx ]
    pred_class = torch.tensor(pred_class)
    # 确保mask和gt_mask的尺寸和数据类型相同
    assert mask.shape == gt_mask.shape, "Mask和GT Mask的尺寸必须相同。"
    # assert mask.dtype == gt_mask.dtype, "Mask和GT Mask的数据类型必须相同。"
    
    # 转换为numpy数组，如果它们还不是
    mask = np.array(mask)
    gt_mask = np.array(gt_mask.cpu())
    
    # 计算Overall Acc
    correct_pixels = np.sum(mask == gt_mask)
    total_pixels = np.prod(mask.shape)
    overall_acc = correct_pixels / total_pixels

    # 计算Mean IoU
    unique_classes = np.unique(gt_mask)
    ious = []
    precisions = []
    recalls = []
    for c in unique_classes:
        # 计算单个类别的IoU
        true_positive = np.sum((mask == c) & (gt_mask == c))
        false_positive = np.sum((mask == c) & (gt_mask != c))
        false_negative = np.sum((mask != c) & (gt_mask == c))
        iou = true_positive / (true_positive + false_positive + false_negative)
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        iou = true_positive / (true_positive + false_positive + false_negative)
        precisions.append(precision)
        recalls.append(recall)
        ious.append(iou)
    mean_iou = np.mean(ious)
    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)
    # iou_thresholds = list[range(50, 95, 5)] * 0.01
    # for i, t in enumerate(iou_thresholds):

    # # 计算Class IoU
    # true_positive = np.sum((mask == pred_class) & (gt_mask == true_class))
    # false_positive = np.sum((mask == pred_class) & (gt_mask != true_class))
    # false_negative = np.sum((mask != pred_class) & (gt_mask == true_class))
    # class_iou = true_positive / (true_positive + false_positive + false_negative)

    # # 计算Class Acc
    # true_positive = np.sum((mask == true_class) & (gt_mask == true_class))
    # false_positive = np.sum((mask == true_class) & (gt_mask != true_class))
    # false_negative = np.sum((mask != true_class) & (gt_mask == true_class))
    # class_acc = true_positive / (true_positive + false_positive)

    return overall_acc, mean_iou,  mean_precision, mean_recall# , class_iou, class_acc

# 示例使用
# 假设mask和gt_mask是已经定义好的B*H*W尺寸的numpy数组
# pred_class = 预测的类别，例如1
# true_class = 真实的类别，例如1
# overall_acc, mean_iou, class_iou, class_acc = compute_metrics(mask, gt_mask, pred_class, true_class)
def top_n_indices(lst, n):
    # 使用enumerate获取元素及其索引，然后根据值进行排序
    sorted_items = sorted(enumerate(lst), key=lambda x: x[1], reverse=True)
    # 提取前n个元素的索引
    top_n_indices = [index for index, value in sorted_items[:n]]
    return top_n_indices

def merge_tensorboard_logs(file1, file2, output_file):
    """
    合并两个TensorBoard日志文件。

    参数:
    file1: 第一个日志文件的路径。
    file2: 第二个日志文件的路径。
    output_file: 输出合并后日志文件的路径。
    """
    # 确保文件存在
    if not os.path.exists(file1) or not os.path.exists(file2):
        raise FileNotFoundError("One of the provided files does not exist.")

    # 读取第一个文件的内容
    with open(file1, 'rb') as f1:
        content1 = f1.read()

    # 读取第二个文件的内容
    with open(file2, 'rb') as f2:
        content2 = f2.read()

    # 合并内容
    merged_content = content1 + content2

    # 写入到新的文件
    with open(output_file, 'wb') as f_out:
        f_out.write(merged_content)

    print(f"文件已合并，合并后的文件位于：{output_file}")

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
    import numpy as np
    from sklearn.decomposition import PCA
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
    if show:
        import matplotlib.pyplot as plt
        from draw_weight import draw_correlation
        x_len = components.shape[-1]
        plt.figure(figsize=(20,10))
        plt.imshow(components, aspect='auto', cmap='viridis')
        plt.colorbar(label='Weight Value')
        plt.title('Weights Matrix Heatmap')
        plt.xlabel('Classes')
        plt.ylabel('Features')
        plt.xticks(np.arange(x_len), [f'{i}' for i in range(x_len)], rotation=90)
        plt.yticks([])
        plt.tight_layout()
        draw_correlation(torch.tensor(components), "./output", cur_epoch=n_components)
        draw_correlation(torch.tensor(new_rows), "./output", cur_epoch=f"new_rows_{n_components}")
        plt.savefig(f"./output/pca_components_{n_components}")
        plt.show()
    return new_rows

def align_bias(bias, new_bias_shape=None):
    # I haven't used it yet
    mean = bias.mean()
    std_dev = bias.std(unbiased=False)
    if new_bias_shape == None:
        new_dimension = bias.shape[0] + 5
    else:
        new_dimension = new_bias_shape
    sampled_tensor = torch.normal(mean=mean, std=std_dev, size=(new_dimension,))
    return sampled_tensor

def spilt_train_val(train_rate, dataset):
    train_size = int(train_rate * len(dataset))
    val_size = len(dataset) - train_size
    # 随机划分数据集
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset

