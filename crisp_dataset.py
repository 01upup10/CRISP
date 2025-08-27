from torch.utils.data import Dataset
from typing import Optional, Callable, Dict
import torch
from torch import Tensor
import json
from collections import OrderedDict
import os
import json

_global_buffer_registration_hooks: Dict[int, Callable] = OrderedDict()
class YTvis2019Dataset(Dataset):
    
    
    def __init__(self, cfg, data_dir, state, mapper, dataset_dict=None, device='cpu'):
        self.data_dir = data_dir
        self.state = state
        pixel_mean = cfg.MODEL.PIXEL_MEAN
        pixel_std = cfg.MODEL.PIXEL_STD
        self.pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1).to(device)
        self.pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1).to(device)
        self.size_divisibility = cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY
        self.num_frames = cfg.INPUT.SAMPLING_FRAME_NUM
        self.device = device
        self.mapper = mapper
        if dataset_dict != None:
            self.dataset_dict = dataset_dict
        if 'train' in state:
            json_path = os.path.join(self.data_dir,state+'.json')
            a = os.path.join(self.data_dir,'train.json')
            self.all_videos = json.load(open(a))['videos']
            self.data = json.load(open(json_path))
            self.videos = self.data['videos']
            self.categories = self.data['categories']
            self.annotations = self.data['annotations']
        if 'test' in state or 'valid' in state:
            json_path = os.path.join(self.data_dir,state+'.json')
            a = os.path.join(self.data_dir,'train.json')
            self.all_videos = json.load(open(a))['videos']
            self.data = json.load(open(json_path))
            self.videos = self.data['videos']
            self.categories = self.data['categories']
            if "split" in state:
                self.annotations = self.data['annotations']
        
    def __len__(self):
        
        return len(self.videos)
    
    def __getitem__(self, index):
        
        video = self.all_videos[index]
        data = self.mapper.__call__(self.dataset_dict[video["id"]-1])
        return data # dict
    
    def __load_current_category__(self):
        dataset = {}
        
        for i in range(40):
            for item in self.annotations:
                if item['category_id'] == self.category_id:
                    dataset.update()
        return dataset
    
    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        gt_instances = []
        targets_per_video = targets
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
    
    def register_buffer(self, name: str, tensor: Optional[Tensor], persistent: bool = True) -> None:
        r"""Add a buffer to the module.

        This is typically used to register a buffer that should not to be
        considered a model parameter. For example, BatchNorm's ``running_mean``
        is not a parameter, but is part of the module's state. Buffers, by
        default, are persistent and will be saved alongside parameters. This
        behavior can be changed by setting :attr:`persistent` to ``False``. The
        only difference between a persistent buffer and a non-persistent buffer
        is that the latter will not be a part of this module's
        :attr:`state_dict`.

        Buffers can be accessed as attributes using given names.

        Args:
            name (str): name of the buffer. The buffer can be accessed
                from this module using the given name
            tensor (Tensor or None): buffer to be registered. If ``None``, then operations
                that run on buffers, such as :attr:`cuda`, are ignored. If ``None``,
                the buffer is **not** included in the module's :attr:`state_dict`.
            persistent (bool): whether the buffer is part of this module's
                :attr:`state_dict`.

        Example::

            >>> # xdoctest: +SKIP("undefined vars")
            >>> self.register_buffer('running_mean', torch.zeros(num_features))

        """
        if persistent is False and isinstance(self, torch.jit.ScriptModule):
            raise RuntimeError("ScriptModule does not support non-persistent buffers")

        if '_buffers' not in self.__dict__:
            raise AttributeError(
                "cannot assign buffer before Module.__init__() call")
        elif not isinstance(name, str):
            raise TypeError(f"buffer name should be a string. Got {torch.typename(name)}")
        elif '.' in name:
            raise KeyError("buffer name can't contain \".\"")
        elif name == '':
            raise KeyError("buffer name can't be empty string \"\"")
        elif hasattr(self, name) and name not in self._buffers:
            raise KeyError(f"attribute '{name}' already exists")
        elif tensor is not None and not isinstance(tensor, torch.Tensor):
            raise TypeError(f"cannot assign '{torch.typename(tensor)}' object to buffer '{name}' "
                            "(torch Tensor or None required)"
                            )
        else:
            for hook in _global_buffer_registration_hooks.values():
                output = hook(self, name, tensor)
                if output is not None:
                    tensor = output
            self._buffers[name] = tensor
            if persistent:
                self._non_persistent_buffers_set.discard(name)
            else:
                self._non_persistent_buffers_set.add(name)



class Subset(torch.utils.data.Dataset):
    """
    Subset of a dataset at specified indices.
    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        transform (callable): way to transform the images and the targets
        target_transform(callable): way to transform the target labels
    """

    def __init__(self, dataset, labels,
                 train=True, valid_on_cur_task_data=False, total_classes=40):
        self.dataset = dataset
        self.valid_on_cur_task_data = valid_on_cur_task_data
        self.labels = labels if train or self.valid_on_cur_task_data else list(range(total_classes))
        subset_idx = [] 
        print('init subset')
        if hasattr(self.dataset, 'annotations'):
            self.pan = False
            annos = self.dataset.annotations
            for i, d in enumerate(annos):
                c_id = d['category_id']-1
                v_id = d["video_id"]
                if c_id in self.labels:
                    subset_idx.append(v_id-1)
        else:
            dataset_dict = self.dataset.dataset_dict
            self.pan = True # panoptic segmentation
            for i, d in enumerate(dataset_dict):
                segments_info = d['segments_info']
                for segment_info in segments_info:
                    c_id = segment_info['category_id']-1
                    if c_id == -1 or c_id == 254:
                        continue
                    v_id = d["id"] 
                    if c_id in self.labels:
                        subset_idx.append(v_id-1)
        subset_idx = sorted(list(set(subset_idx))) 
        self.subset_ids = subset_idx 

    def __getitem__(self, idx):
        samples = self.dataset[self.subset_ids[idx]]

        bkg_id = self.labels[-1]+1
        for j in range(len(samples["instances"])):
            gt_classes = samples["instances"][j].gt_classes

            for i, c in enumerate(gt_classes):
                if c not in self.labels:
                    samples["instances"][j].gt_classes[i] = torch.tensor(bkg_id)
                    gt_masks = samples["instances"][j].gt_masks.tensor[i]
                    samples["instances"][j].gt_masks.tensor[i] = torch.where(gt_masks, torch.tensor(False), torch.tensor(False))
        return samples

    def __len__(self):
        return len(self.subset_ids)
    
class YTVIS2019SegmentationIncremental(Dataset):
    def __init__(self,
                 cfg,
                 data_dir,
                 state,
                 mapper,
                 dataset_dict,
                 train=True,
                 labels=None,
                 valid_on_cur_task_data=False,
                 total_classes=40
                 ):
        full_voc = YTvis2019Dataset(cfg, data_dir, state, mapper, dataset_dict, device='cuda')
        if labels is not None:
            self.dataset = Subset(full_voc, labels, train=train, 
                                  valid_on_cur_task_data=valid_on_cur_task_data, total_classes=total_classes)
        else:
            self.dataset = full_voc

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)
