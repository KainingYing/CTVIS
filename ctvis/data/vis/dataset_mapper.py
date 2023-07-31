import copy
import logging
import random
import numpy as np
from typing import List, Union
import torch
from torch import distributed as dist

from detectron2.config import configurable
from detectron2.structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Instances,
)

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

from .augmentation import build_augmentation
from ..pseudo_video import build_pseudo_augmentation, annotations_to_instances, convert_coco_poly_to_mask

__all__ = ["YTVISDatasetMapper", ]


def get_dist_info():
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def filter_empty_instances(instances, by_box=True, by_mask=True, box_threshold=1e-5):
    """
    Filter out empty instances in an `Instances` object.

    Args:
        instances (Instances):
        by_box (bool): whether to filter out instances with empty boxes
        by_mask (bool): whether to filter out instances with empty masks
        box_threshold (float): minimum width and height to be considered non-empty

    Returns:
        Instances: the filtered instances.
    """
    assert by_box or by_mask
    r = []
    if by_box:
        r.append(instances.gt_boxes.nonempty(threshold=box_threshold))
    if instances.has("gt_masks") and by_mask:
        r.append(instances.gt_masks.nonempty())

    if not r:
        return instances
    m = r[0]
    for x in r[1:]:
        m = m & x

    instances.gt_ids[~m] = -1
    return instances


def _get_dummy_anno(num_classes):
    return {
        "iscrowd": 0,
        "category_id": num_classes,
        "id": -1,
        "mask_id": -1,
        "frame_id": -1,
        "bbox": np.array([0, 0, 0, 0]),
        "bbox_mode": BoxMode.XYXY_ABS,
        "segmentation": [np.array([0.0] * 6)],
        "generate": False
    }


def ytvis_annotations_to_instances(annos, image_size):
    """
    Create an :class:`Instances` object used by the models,
    from instance annotations in the dataset dict.

    Args:
        annos (list[dict]): a list of instance annotations in one image, each
            element for one instance.
        image_size (tuple): height, width

    Returns:
        Instances:
            It will contain fields "gt_boxes", "gt_classes", "gt_ids",
            "gt_masks", if they can be obtained from `annos`.
            This is the format that builtin models expect.
    """
    boxes = [BoxMode.convert(
        obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS) for obj in annos]
    target = Instances(image_size)
    target.gt_boxes = Boxes(boxes)

    classes = [int(obj["category_id"]) for obj in annos]
    classes = torch.tensor(classes, dtype=torch.int64)
    target.gt_classes = classes

    ids = [int(obj["id"]) for obj in annos]
    ids = torch.tensor(ids, dtype=torch.int64)
    target.gt_ids = ids

    if len(annos) and "segmentation" in annos[0]:
        segms = [obj["segmentation"] for obj in annos]
        masks = []
        for segm in segms:
            assert segm.ndim == 2, "Expect segmentation of 2 dimensions, got {}.".format(
                segm.ndim
            )
            # mask array
            masks.append(segm)
        # torch.from_numpy does not support array with negative stride.
        masks = BitMasks(
            torch.stack([torch.from_numpy(np.ascontiguousarray(x))
                        for x in masks])
        )
        target.gt_masks = masks

    return target


class YTVISDatasetMapper:
    """
    A callable which takes a dataset dict in YouTube-VIS Dataset format,
    and map it into a format used by the model.
    """

    @configurable
    def __init__(
            self,
            is_train: bool,
            *,
            augmentations: List[Union[T.Augmentation, T.Transform]],
            augmentations_with_crop: List[Union[T.Augmentation, T.Transform]],
            image_format: str,
            use_instance_mask: bool = False,
            sampling_frame_num: int = 2,
            sampling_frame_range: int = 5,
            sampling_frame_shuffle: bool = False,
            num_classes: int = 40,
            image_mode=False,
            pseudo_augs=None
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            use_instance_mask: whether to process instance segmentation annotations, if available
        """
        # fmt: off
        self.is_train = is_train

        self.augmentations = T.AugmentationList(augmentations)
        if augmentations_with_crop is not None:
            self.augmentations_with_crop = T.AugmentationList(augmentations_with_crop)
        else:
            self.augmentations_with_crop = None
        self.image_format = image_format
        self.use_instance_mask = use_instance_mask
        self.sampling_frame_num = sampling_frame_num
        self.sampling_frame_range = sampling_frame_range
        self.sampling_frame_shuffle = sampling_frame_shuffle
        self.num_classes = num_classes
        self.image_mode = image_mode
        self.pseudo_augs = T.AugmentationList(pseudo_augs)
        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(
            f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")
        if self.augmentations_with_crop is not None:
            logger.info(
                f"[DatasetMapper] Augmentations With Crop used in {mode}: {augmentations_with_crop}")

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        # augs = build_augmentation(cfg, is_train)
        image_mode = cfg.INPUT.IMAGE_MODE

        sampling_frame_num = cfg.INPUT.SAMPLING_FRAME_NUM
        sampling_frame_range = cfg.INPUT.SAMPLING_FRAME_RANGE
        sampling_frame_shuffle = cfg.INPUT.SAMPLING_FRAME_SHUFFLE

        if cfg.INPUT.CROP.ENABLED and is_train:
            augs_with_crop, augs = build_augmentation(cfg, is_train)
        else:
            augs_with_crop = None
            augs = build_augmentation(cfg, is_train)

        pseudo_augs = build_pseudo_augmentation(cfg, is_train)

        ret = {
            "is_train": is_train,
            "image_mode": image_mode,
            "augmentations": augs,
            "augmentations_with_crop": augs_with_crop,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "sampling_frame_num": sampling_frame_num,
            "sampling_frame_range": sampling_frame_range,
            "sampling_frame_shuffle": sampling_frame_shuffle,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "pseudo_augs": pseudo_augs
        }

        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one video, in YTVIS Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """

        if self.is_train and self.image_mode:
            return self.prepare_pseudo_video(dataset_dict)
        else:
            return self.prepare_video(dataset_dict)

    def prepare_video(self, dataset_dict):
        # TODO consider examining below deepcopy as it costs huge amount of computations.
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below  # noqa

        video_length = dataset_dict["length"]
        if self.is_train:
            ref_frame = random.randrange(video_length)

            start_idx = max(0, ref_frame - self.sampling_frame_range)
            end_idx = min(video_length, ref_frame +
                          self.sampling_frame_range + 1)

            selected_idx = np.random.choice(
                np.array(list(range(start_idx, ref_frame)) +
                         list(range(ref_frame + 1, end_idx))),
                self.sampling_frame_num - 1).tolist()
            # selected_idx = random.sample(set(range(start_idx, end_idx)) - set([ref_frame]),
            #                              self.sampling_frame_num - 1)  # noqa
            selected_idx = selected_idx + [ref_frame]
            selected_idx = sorted(selected_idx)
            if self.sampling_frame_shuffle:
                random.shuffle(selected_idx)
        else:
            selected_idx = range(video_length)

        video_annos = dataset_dict.pop("annotations", None)
        file_names = dataset_dict.pop("file_names", None)

        if self.is_train or video_annos is not None:
            _ids = set()
            for frame_idx in selected_idx:
                _ids.update([anno["id"] for anno in video_annos[frame_idx]])
            ids = dict()
            for i, _id in enumerate(_ids):
                ids[_id] = i

        dataset_dict["video_len"] = len(video_annos)
        dataset_dict["frame_idx"] = list(selected_idx)
        dataset_dict["image"] = []
        dataset_dict["instances"] = []
        dataset_dict["file_names"] = []

        if self.augmentations_with_crop is not None and self.is_train:
            if np.random.rand() > 0.5:  # random
                augmentations = self.augmentations_with_crop
            else:
                augmentations = self.augmentations
        else:
            augmentations = self.augmentations

        for frame_idx in selected_idx:
            dataset_dict["file_names"].append(file_names[frame_idx])

            # Read image
            image = utils.read_image(
                file_names[frame_idx], format=self.image_format)
            utils.check_image_size(dataset_dict, image)

            aug_input = T.AugInput(image)
            transforms = augmentations(aug_input)
            image = aug_input.image

            image_shape = image.shape[:2]  # h, w
            # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
            # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
            # Therefore it's important to use torch.Tensor.
            dataset_dict["image"].append(torch.as_tensor(
                np.ascontiguousarray(image.transpose(2, 0, 1))))

            if (video_annos is None) and (not self.is_train):
                continue

            # NOTE copy() is to prevent annotations getting changed from applying augmentations
            _frame_annos = []
            for anno in video_annos[frame_idx]:
                _anno = {}
                for k, v in anno.items():
                    _anno[k] = copy.deepcopy(v)
                _frame_annos.append(_anno)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape)
                for obj in _frame_annos
                if obj.get("iscrowd", 0) == 0
            ]
            sorted_annos = [_get_dummy_anno(self.num_classes)
                            for _ in range(len(ids))]

            for _anno in annos:
                idx = ids[_anno["id"]]
                sorted_annos[idx] = _anno
            _gt_ids = [_anno["id"] for _anno in sorted_annos]
            _generate_flags = [_anno["generate"] for _anno in sorted_annos]
            # _

            # _gt_mask_ids = [_anno["mask_id"] for _anno in sorted_annos]
            # _gt_frame_ids = [_anno["frame_id"] for _anno in sorted_annos]
            instances = utils.annotations_to_instances(
                sorted_annos, image_shape, mask_format="bitmask")

            instances.gt_ids = torch.tensor(_gt_ids)
            # instances.gt_mask_ids = torch.tensor(_gt_mask_ids)
            # instances.gt_frame_ids = torch.tensor(_gt_frame_ids)
            if instances.has("gt_masks"):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
                instances = filter_empty_instances(instances)
            else:
                instances.gt_masks = BitMasks(torch.empty((0, *image_shape)))

            coco_flag = [False] * len(_gt_ids)

            instances.coco_flag = torch.tensor(coco_flag)
            instances.generate_flag = torch.tensor(_generate_flags)
            dataset_dict["instances"].append(instances)

        return dataset_dict

    def prepare_pseudo_video(self, dataset_dict):
        # it will be modified by code below
        dataset_dict = copy.deepcopy(dataset_dict)

        video_length = dataset_dict["length"]
        frame_idx = random.randrange(video_length)

        video_annos = dataset_dict.pop("annotations", None)
        file_names = dataset_dict.pop("file_names", None)

        file_name = file_names[frame_idx]
        image_anno = video_annos[frame_idx]
        original_image = utils.read_image(file_name, format=self.image_format)

        dataset_dict["image"] = []
        dataset_dict["instances"] = []
        dataset_dict["file_names"] = [file_name] * self.sampling_frame_num

        for i in range(self.sampling_frame_num):
            utils.check_image_size(dataset_dict, original_image)

            aug_input = T.AugInput(original_image)
            transforms = self.pseudo_augs(aug_input)
            image = aug_input.image
            image_shape = image.shape[:2]  # h, w

            dataset_dict["image"].append(torch.as_tensor(
                np.ascontiguousarray(image.transpose(2, 0, 1))))

            if (image_anno is None) or (not self.is_train):
                continue

            _img_annos = []
            for anno in image_anno:
                _anno = {}
                for k, v in anno.items():
                    _anno[k] = copy.deepcopy(v)
                _img_annos.append(_anno)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape)
                for obj in _img_annos
                if obj.get("iscrowd", 0) == 0
            ]
            _gt_ids = list(range(len(annos)))
            for idx in range(len(annos)):
                if len(annos[idx]["segmentation"]) == 0:
                    annos[idx]["segmentation"] = [np.array([0.0] * 6)]

            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format="bitmask")

            instances.gt_ids = torch.tensor(_gt_ids)
            # instances.gt_mask_ids = torch.tensor(_gt_mask_ids)
            # instances.gt_frame_ids = torch.tensor(_gt_frame_ids)
            if instances.has("gt_masks"):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
                instances = filter_empty_instances(instances)
            else:
                instances.gt_masks = BitMasks(torch.empty((0, *image_shape)))

            coco_flag = [False] * len(_gt_ids)
            instances.coco_flag = torch.tensor(coco_flag)

            dataset_dict["instances"].append(instances)

        return dataset_dict
