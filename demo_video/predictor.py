# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# reference: https://github.com/sukjunhwang/IFC/blob/master/projects/IFC/demo/predictor.py
import atexit
import bisect
import multiprocessing as mp
from collections import deque
import cv2
import torch

from .visualizer import TrackVisualizer

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.structures import Instances
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode

# import torch
from collections import defaultdict

class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.parallel = parallel
        if parallel:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        else:
            self.predictor = VideoPredictor(cfg)

    def run_on_video(self, frames):
        """
        Args:
            frames (List[np.ndarray]): a list of images of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.
        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        predictions = self.predictor(frames)

        image_size = predictions["image_size"]
        pred_scores = predictions["pred_scores"]
        pred_labels = predictions["pred_labels"]
        pred_masks = predictions["pred_masks"]

        frame_masks = list(zip(*pred_masks))
        total_vis_output = []
        for frame_idx in range(len(frames)):
            frame = frames[frame_idx][:, :, ::-1]
            visualizer = TrackVisualizer(frame, self.metadata, instance_mode=self.instance_mode)
            ins = Instances(image_size)
            if len(pred_scores) > 0:
                ins.scores = pred_scores
                ins.pred_classes = pred_labels
                ins.pred_masks = torch.stack(frame_masks[frame_idx], dim=0)

            vis_output = visualizer.draw_instance_predictions(predictions=ins)
            total_vis_output.append(vis_output)

        return predictions, total_vis_output


class VideoPredictor(DefaultPredictor):
    """
    Create a simple end-to-end predictor with the given config that runs on
    single device for a single input image.
    Compared to using the model directly, this class does the following additions:
    1. Load checkpoint from `cfg.MODEL.WEIGHTS`.
    2. Always take BGR image as the input and apply conversion defined by `cfg.INPUT.FORMAT`.
    3. Apply resizing defined by `cfg.INPUT.{MIN,MAX}_SIZE_TEST`.
    4. Take one input image and produce a single output, instead of a batch.
    If you'd like to do anything more fancy, please refer to its source code
    as examples to build and use the model manually.
    Attributes:
        metadata (Metadata): the metadata of the underlying dataset, obtained from
            cfg.DATASETS.TEST.
    Examples:
    ::
        pred = DefaultPredictor(cfg)
        inputs = cv2.imread("input.jpg")
        outputs = pred(inputs)
    """
    def __call__(self, frames):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).
        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            input_frames = []
            for original_image in frames:
                # Apply pre-processing to image.
                if self.input_format == "RGB":
                    # whether the model expects BGR inputs or RGB
                    original_image = original_image[:, :, ::-1]
                height, width = original_image.shape[:2]
                image = self.aug.get_transform(original_image).apply_image(original_image)
                image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
                input_frames.append(image)

            inputs = {"image": input_frames, "height": height, "width": width}
            predictions = self.model([inputs])
            return predictions


class AsyncPredictor:
    """
    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    Because rendering the visualization takes considerably amount of time,
    this helps improve throughput when rendering videos.
    """

    class _StopToken:
        pass

    class _PredictWorker(mp.Process):
        def __init__(self, cfg, task_queue, result_queue):
            self.cfg = cfg
            self.task_queue = task_queue
            self.result_queue = result_queue
            super().__init__()

        def run(self):
            predictor = VideoPredictor(self.cfg)

            while True:
                task = self.task_queue.get()
                if isinstance(task, AsyncPredictor._StopToken):
                    break
                idx, data = task
                result = predictor(data)
                self.result_queue.put((idx, result))

    def __init__(self, cfg, num_gpus: int = 1):
        """
        Args:
            cfg (CfgNode):
            num_gpus (int): if 0, will run on CPU
        """
        num_workers = max(num_gpus, 1)
        self.task_queue = mp.Queue(maxsize=num_workers * 3)
        self.result_queue = mp.Queue(maxsize=num_workers * 3)
        self.procs = []
        for gpuid in range(max(num_gpus, 1)):
            cfg = cfg.clone()
            cfg.defrost()
            cfg.MODEL.DEVICE = "cuda:{}".format(gpuid) if num_gpus > 0 else "cpu"
            self.procs.append(
                AsyncPredictor._PredictWorker(cfg, self.task_queue, self.result_queue)
            )

        self.put_idx = 0
        self.get_idx = 0
        self.result_rank = []
        self.result_data = []

        for p in self.procs:
            p.start()
        atexit.register(self.shutdown)

    def put(self, image):
        self.put_idx += 1
        self.task_queue.put((self.put_idx, image))

    def get(self):
        self.get_idx += 1  # the index needed for this request
        if len(self.result_rank) and self.result_rank[0] == self.get_idx:
            res = self.result_data[0]
            del self.result_data[0], self.result_rank[0]
            return res

        while True:
            # make sure the results are returned in the correct order
            idx, res = self.result_queue.get()
            if idx == self.get_idx:
                return res
            insert = bisect.bisect(self.result_rank, idx)
            self.result_rank.insert(insert, idx)
            self.result_data.insert(insert, res)

    def __len__(self):
        return self.put_idx - self.get_idx

    def __call__(self, image):
        self.put(image)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(AsyncPredictor._StopToken())

    @property
    def default_buffer_size(self):
        return len(self.procs) * 5

# def _create_text_labels(classes, scores, class_names, is_crowd=None):
#     """
#     Args:
#         classes (list[int] or None):
#         scores (list[float] or None):
#         class_names (list[str] or None):
#         is_crowd (list[bool] or None):

#     Returns:
#         list[str] or None
#     """
#     labels = None
#     if classes is not None:
#         if class_names is not None and len(class_names) > 0:
#             labels = [class_names[i] for i in classes]
#         else:
#             labels = [str(i) for i in classes]
#     if scores is not None:
#         if labels is None:
#             labels = ["{:.0f}%".format(s * 100) for s in scores]
#         else:
#             labels = ["{} {:.0f}%".format(l, s * 100) for l, s in zip(labels, scores)]
#     if labels is not None and is_crowd is not None:
#         labels = [l + ("|crowd" if crowd else "") for l, crowd in zip(labels, is_crowd)]
#     return labels
from detectron2.utils.visualizer import ColorMode, GenericMask, Visualizer, _create_text_labels
class VisualizationDemo2(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False, predictor=None):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.parallel = parallel
        if parallel:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        elif predictor==None:
            self.predictor = VideoPredictor(cfg)
        else: 
            self.predictor = predictor

    def run_on_video(self, frames, inputs):
        """
        Args:
            frames (List[np.ndarray]): a list of images of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.
        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        predictions = self.predictor(inputs)

        # 过滤重复的mask
        import numpy as np
        def filter_masks(predictions, iou_threshold=0.5, iom_threshold=0.6):
            pred_scores = predictions["pred_scores"]
            pred_labels = predictions["pred_labels"]
            pred_masks = predictions["pred_masks"]
            decoder_feat = predictions["decoder_feat"]
            topk_indices = predictions["topk_indices"]
            # 按置信度分数降序排列
            sorted_indices = np.argsort(pred_scores)[::-1]
            sorted_masks = [pred_masks[i] for i in sorted_indices]
            sorted_labels = [pred_labels[i] for i in sorted_indices]
            sorted_scores = [pred_scores[i] for i in sorted_indices]
            
            sorted_topkID = [topk_indices[i] for i in sorted_indices]

            keep_masks = []
            keep_labels = []
            keep_scores = []

            keep_topkID = []

            n = len(sorted_masks)
            # keeps = [True] * n
            for i in range(n):
                current_mask = sorted_masks[i]
                current_label = sorted_labels[i]
                current_score = sorted_scores[i]

                current_topkID = sorted_topkID[i]
                
                keep = True
                # if current_score > 0.9:
                #     keep = True
                # else:
                #     keep = False
                # 检查与已保留掩码的同类别IoU
                for k in range(len(keep_masks)):
                    # if keep_labels[k] != current_label:
                    #     continue
                    kept_mask = keep_masks[k]
                    kept_score = keep_scores[k]
                    # 计算IoU
                    intersection = np.logical_and(current_mask, kept_mask)
                    union = np.logical_or(current_mask, kept_mask)
                    iou = intersection.sum() / (union.sum()+1e-5)

                    cur_area = current_mask.sum()
                    keep_area = kept_mask.sum()
                    x = intersection.sum()
                    iom =  x / (min(cur_area, keep_area)+1e-5)

                    if iou >= iou_threshold or iom >= iom_threshold:
                        keep = False
                        break
                if keep:
                    keep_masks.append(current_mask)
                    keep_labels.append(current_label)
                    keep_scores.append(current_score)

                    keep_topkID.append(current_topkID)

            new_predictions = {
                "image_size": predictions["image_size"],
                "pred_scores": keep_scores,
                "pred_labels": keep_labels,
                "pred_masks": keep_masks,
                "decoder_feat": decoder_feat,
                "topkID": keep_topkID,
                "targets": predictions["targets"],
                "att_map": predictions['att_map']
            }
            new_predictions
            
            return new_predictions
        
        predictions = filter_masks(predictions)

        image_size = predictions["image_size"]
        pred_scores = predictions["pred_scores"]
        pred_labels = predictions["pred_labels"]
        pred_masks = predictions["pred_masks"]
        # decoder_feat = predictions["decoder_feat"
        targets = predictions["targets"][0]
        boxes = None
        scores = None
        classes = [i.detach().cpu() for i in targets["labels"]]
        labels = _create_text_labels(classes, scores, self.metadata.get("thing_classes", None))
        masks = [targets["masks"][i].detach().cpu().bool() for i in range(len(labels))]
        gt_masks = list(zip(*masks))
        
        
        frame_masks = list(zip(*pred_masks))
        total_vis_output = []
        total_gt_outout = []
        for frame_idx in range(len(frames)):
            frame = frames[frame_idx][:, :, ::-1]
            visualizer = TrackVisualizer(frame, self.metadata, instance_mode=self.instance_mode)
            visualizer_gt = TrackVisualizer(frame, self.metadata, instance_mode=self.instance_mode)
            ins = Instances(image_size)
            if len(pred_scores) > 0:
                ins.scores = pred_scores
                ins.pred_classes = pred_labels
                ins.pred_masks = torch.stack(frame_masks[frame_idx], dim=0)

            # target_fields = per_image["instances"].get_fields()
            # labels = [metadata.thing_classes[i] for i in target_fields["gt_classes"]]
            # vis = visualizer.overlay_instances(
            #     labels=labels,
            #     boxes=target_fields.get("gt_boxes", None),
            #     masks=target_fields.get("gt_masks", None),
            #     keypoints=target_fields.get("gt_keypoints", None),
            # )
            # output(vis, str(per_image["image_id"]) + ".jpg")
            cur_masks = torch.stack(gt_masks[frame_idx], dim=0).numpy()
            cur_masks = np.asarray(cur_masks)
            masks = [GenericMask(x, visualizer.output.height, visualizer.output.width) for x in cur_masks]
            colors = [
            visualizer_gt._jitter([x / 255 for x in self.metadata.thing_colors[c]], id) for id, c in enumerate(classes)
            ]
            alpha = 0.5
            gt_outout = visualizer_gt.overlay_instances(
                    masks=cur_masks,
                    boxes=boxes,
                    labels=labels,
                    assigned_colors=colors,
                    alpha=alpha,
                )
            vis_output = visualizer.draw_instance_predictions(predictions=ins)
            total_vis_output.append(vis_output)
            total_gt_outout.append(gt_outout)
            torch.cuda.empty_cache()
        return predictions, total_vis_output, total_gt_outout
    
    def mask_iou(self, mask1, mask2):
        """计算两个mask之间的IoU"""
        intersection = torch.logical_and(mask1, mask2).sum().item()
        union = torch.logical_or(mask1, mask2).sum().item()
        return intersection / union if union != 0 else 0.0

    def nms_masks(self, predictions, iou_threshold=0.5):
        """
        对同一类别的预测应用NMS。
        predictions: 当前类别的预测列表，每个元素包含'mask', 'score'。
        返回保留的预测索引。
        """
        if not predictions:
            return []
        
        # 按score降序排序
        sorted_indices = sorted(
            range(len(predictions)),
            key=lambda i: predictions[i]['score'],
            reverse=True
        )
        
        keep = []
        while sorted_indices:
            current_idx = sorted_indices.pop(0)
            keep.append(current_idx)
            
            # 筛选出需要保留的索引
            filtered = []
            for idx in sorted_indices:
                current_mask = predictions[current_idx]['mask']
                other_mask = predictions[idx]['mask']
                iou = self.mask_iou(current_mask, other_mask)
                if iou <= iou_threshold:
                    filtered.append(idx)
            sorted_indices = filtered
        
        return keep

    # # 假设输入的predictions为top10的预测列表，每个元素包含'score', 'mask', 'label'
    # predictions = [
    #     {'score': 0.9, 'mask': torch.tensor(...), 'label': 'cat'},
    #     {'score': 0.8, 'mask': torch.tensor(...), 'label': 'dog'},
    #     # ... 其他预测
    # ]

    # # 按类别分组
    # grouped = defaultdict(list)
    # for pred in predictions:
    #     grouped[pred['label']].append(pred)

    # # 处理每个类别的预测
    # final_predictions = []
    # for label, preds in grouped.items():
    #     # 执行NMS
    #     keep_indices = nms_masks(preds, iou_threshold=0.5)
    #     # 保留结果
    #     for idx in keep_indices:
    #         final_predictions.append(preds[idx])

    # # 最终final_predictions即为处理后的结果，每个实例在同一类别下仅保留一个mask