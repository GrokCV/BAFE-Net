import copy
import inspect
import math
import os
import cv2
import glob
import mmcv
import torch
import warnings
import skimage
import skimage.measure
import skimage.io
import numpy as np
import os.path as osp
from PIL import Image
from numpy import random
from typing import List, Optional, Sequence, Tuple, Union
from mmcv.image.geometric import _scale_size
from mmcv.transforms import BaseTransform
from mmcv.transforms import Pad
from mmcv.transforms import RandomFlip
from mmcv.transforms import Resize
from mmcv.transforms.utils import avoid_cache_randomness, cache_randomness
from mmengine.dataset import BaseDataset
from mmengine.utils import is_str
from mmdet.structures.bbox import HorizontalBoxes, autocast_box_type, get_box_type
from mmdet.structures.mask import BitmapMasks, PolygonMasks
from mmdet.utils import log_img_scale
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
from lxml import etree
from deepir.registry import TRANSFORMS


@TRANSFORMS.register_module()
class CopyPaste(BaseTransform):
    """Simple Copy-Paste is a Strong Data Augmentation Method for Instance
    Segmentation The simple copy-paste transform steps are as follows:

    1. The destination image is already resized with aspect ratio kept,
       cropped and padded.
    2. Randomly select a source image, which is also already resized
       with aspect ratio kept, cropped and padded in a similar way
       as the destination image.
    3. Randomly select some objects from the source image.
    4. Paste these source objects to the destination image directly,
       due to the source and destination image have the same size.
    5. Update object masks of the destination image, for some origin objects
       may be occluded.
    6. Generate bboxes from the updated destination masks and
       filter some objects which are totally occluded, and adjust bboxes
       which are partly occluded.
    7. Append selected source bboxes, masks, and labels.

    Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_bboxes_labels (np.int64) (optional)
    - gt_ignore_flags (bool) (optional)
    - gt_masks (BitmapMasks) (optional)

    Modified Keys:

    - img
    - gt_bboxes (optional)
    - gt_bboxes_labels (optional)
    - gt_ignore_flags (optional)
    - gt_masks (optional)

    Args:
        max_num_pasted (int): The maximum number of pasted objects.
            Defaults to 100.
        bbox_occluded_thr (int): The threshold of occluded bbox.
            Defaults to 10.
        mask_occluded_thr (int): The threshold of occluded mask.
            Defaults to 300.
        selected (bool): Whether select objects or not. If select is False,
            all objects of the source image will be pasted to the
            destination image.
            Defaults to True.
        paste_by_box (bool): Whether use boxes as masks when masks are not
            available.
            Defaults to False.
        mode: 1-原始copypaste
              2-成簇的原始copypaste
              3-高斯copypaste
              4-成簇的高斯copypaste
    """

    def __init__(
            self,
            max_num_pasted: int = 100,
            bbox_occluded_thr: int = 10,
            mask_occluded_thr: int = 300,
            selected: bool = True,
            paste_by_box: bool = False,
            mode: int = 1,
    ) -> None:
        self.max_num_pasted = max_num_pasted
        self.bbox_occluded_thr = bbox_occluded_thr
        self.mask_occluded_thr = mask_occluded_thr
        self.selected = selected
        self.paste_by_box = paste_by_box
        self.mode = mode

    @cache_randomness
    def get_indexes(self, dataset: BaseDataset) -> int:
        """Call function to collect indexes.s.

        Args:
            dataset (:obj:`MultiImageMixDataset`): The dataset.
        Returns:
            list: Indexes.
        """
        return random.randint(0, len(dataset))

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        """Transform function to make a copy-paste of image.

        Args:
            results (dict): Result dict.
        Returns:
            dict: Result dict with copy-paste transformed.
        """

        assert 'mix_results' in results
        num_images = len(results['mix_results'])
        assert num_images == 1, \
            f'CopyPaste only supports processing 2 images, got {num_images}'
        if self.selected:
            selected_results = self._select_object(results['mix_results'][0])
        else:
            selected_results = results['mix_results'][0]
        
        if self.mode == 1: #原始copypaste
            return self._copy_paste_1(results, selected_results)
        elif self.mode == 2: #成簇的原始copypaste
            return self._copy_paste_2(results, selected_results)
        elif self.mode == 3: #高斯copypaste
            return self._copy_paste_3(results, selected_results)
        elif self.mode == 4: #成簇的高斯copypaste
            return self._copy_paste_4(results, selected_results)
        

    @cache_randomness
    def _get_selected_inds(self, num_bboxes: int) -> np.ndarray:
        max_num_pasted = min(num_bboxes + 1, self.max_num_pasted)
        num_pasted = np.random.randint(0, max_num_pasted)
        return np.random.choice(num_bboxes, size=num_pasted, replace=False)

    def get_gt_masks(self, results: dict) -> BitmapMasks:
        """Get gt_masks originally or generated based on bboxes.

        If gt_masks is not contained in results,
        it will be generated based on gt_bboxes.
        Args:
            results (dict): Result dict.
        Returns:
            BitmapMasks: gt_masks, originally or generated based on bboxes.
        """
        if results.get('gt_masks', None) is not None:
            if self.paste_by_box:
                warnings.warn('gt_masks is already contained in results, '
                              'so paste_by_box is disabled.')
            return results['gt_masks']
        else:
            if not self.paste_by_box:
                raise RuntimeError('results does not contain masks.')
            return results['gt_bboxes'].create_masks(results['img'].shape[:2])

    def _select_object(self, results: dict) -> dict:
        """Select some objects from the source results."""
        bboxes = results['gt_bboxes']
        labels = results['gt_bboxes_labels']
        masks = self.get_gt_masks(results)
        ignore_flags = results['gt_ignore_flags']

        selected_inds = self._get_selected_inds(bboxes.shape[0])

        selected_bboxes = bboxes[selected_inds]
        selected_labels = labels[selected_inds]
        selected_masks = masks[selected_inds]
        selected_ignore_flags = ignore_flags[selected_inds]

        results['gt_bboxes'] = selected_bboxes
        results['gt_bboxes_labels'] = selected_labels
        results['gt_masks'] = selected_masks
        results['gt_ignore_flags'] = selected_ignore_flags
        return results

    def _copy_paste_1(self, dst_results: dict, src_results: dict) -> dict:
        """CopyPaste transform function.

        Args:
            dst_results (dict): Result dict of the destination image.
            src_results (dict): Result dict of the source image.
        Returns:
            dict: Updated result dict.
        """
        dst_img = dst_results['img']
        dst_bboxes = dst_results['gt_bboxes']
        dst_labels = dst_results['gt_bboxes_labels']
        dst_masks = self.get_gt_masks(dst_results)
        dst_ignore_flags = dst_results['gt_ignore_flags']

        src_img = src_results['img']
        src_bboxes = src_results['gt_bboxes']
        src_labels = src_results['gt_bboxes_labels']
        src_masks = src_results['gt_masks']
        src_ignore_flags = src_results['gt_ignore_flags']

        if len(src_bboxes) == 0:
            return dst_results

        # generate new mask and bboxes
        updated_src_mask = np.zeros_like(src_masks.masks)

        # 读取原图像mask
        img_basename = osp.basename(dst_results['img_path'])
        dst_mask_path = osp.join("data/SIRSTdevkit/SIRST/BinaryMask", img_basename.replace('.png', '_pixels0.png'))
        mask_img = Image.open(dst_mask_path)
        dst_mask_gt = np.array(mask_img)
        dst_mask_gt = cv2.resize(dst_mask_gt, (512, 512), interpolation=cv2.INTER_NEAREST)
        if dst_results['flip']:
            dst_mask_gt = cv2.flip(dst_mask_gt, 1) # 水平翻转
        target_mask = dst_mask_gt

        for i, src_bbox in enumerate(src_bboxes):
            x1, x2 = int(src_bbox.tensor[0, 0].item()), int(src_bbox.tensor[0, 2].item())
            y1, y2 = int(src_bbox.tensor[0, 1].item()), int(src_bbox.tensor[0, 3].item())
            width = x2 - x1
            height = y2 - y1

            if width >= 10 or height >= 10 :
                width = 7
                height = 7

            # 随机选择位置进行粘贴
            while True:
                y = np.random.randint(0, target_mask.shape[1])
                x = np.random.randint(0, target_mask.shape[0])

                target_overlap_sky = target_mask[x-1:x+height+1, y-1:y+width+1].any()
                if not target_overlap_sky:
                    target_mask[x:x + height, y:y + width] = 255
                    updated_src_mask[i][x:x + height, y:y + width] = 1
                    break
                else:
                    continue

        updated_src_masks = BitmapMasks(updated_src_mask, updated_src_mask.shape[1], updated_src_mask.shape[2])
        updated_src_bboxes = updated_src_masks.get_bboxes(type(src_bboxes))

        # generate new img
        gray_img = src_img[:, :, 0]
        new_src_img_ = gray_img.astype(np.float32)

        img_count_mask = np.zeros_like(gray_img, dtype=float)
        for (bbox, updated_bbox) in zip(src_bboxes, updated_src_bboxes):
            x1, x2 = int(bbox.tensor[0, 0].item()), int(bbox.tensor[0, 2].item())
            y1, y2 = int(bbox.tensor[0, 1].item()), int(bbox.tensor[0, 3].item())
            new_x1, new_x2 = int(updated_bbox.tensor[0, 0].item()), int(updated_bbox.tensor[0, 2].item())
            new_y1, new_y2 = int(updated_bbox.tensor[0, 1].item()), int(updated_bbox.tensor[0, 3].item())
            if new_src_img_[new_y1:new_y2, new_x1:new_x2].shape==(0,0):
                continue
            
            choose_src = gray_img[y1:y2, x1:x2]
            # 大目标缩小
            height = y2 - y1
            width = x2 - x1
            if height >= 10 or width >= 10 :
                choose_src = cv2.resize(choose_src, (7, 7))
                
            new_h = new_y2-new_y1
            new_w = new_x2-new_x1
            for i in range(new_h):
                for j in range(new_w):
                    new_src_img_[new_y1 + i, new_x1 + j] = choose_src[i, j]

            # new_src_img_[new_y1:new_y2, new_x1:new_x2] = choose_src
            img_count_mask[new_y1:new_y2, new_x1:new_x2] += 1
        img_non_zero_count_indices = img_count_mask > 0
        new_src_img_[img_non_zero_count_indices] /= img_count_mask[img_non_zero_count_indices]
        new_src_img = new_src_img_.astype(np.uint8)
        new_src_img = np.stack([new_src_img, new_src_img, new_src_img], axis=-1)

        # update masks and generate bboxes from updated masks
        composed_mask = np.where(np.any(updated_src_masks.masks, axis=0), 1, 0)
        updated_dst_masks = self._get_updated_masks(dst_masks, composed_mask)
        updated_dst_bboxes = updated_dst_masks.get_bboxes(type(dst_bboxes))
        assert len(updated_dst_bboxes) == len(updated_dst_masks)

        # filter totally occluded objects
        l1_distance = (updated_dst_bboxes.tensor - dst_bboxes.tensor).abs()
        bboxes_inds = (l1_distance <= self.bbox_occluded_thr).all(dim=-1).numpy()
        masks_inds = updated_dst_masks.masks.sum(axis=(1, 2)) > self.mask_occluded_thr
        valid_inds = bboxes_inds | masks_inds

        # Paste source objects to destination image directly
        img = dst_img * (1 - composed_mask[..., np.newaxis]) + new_src_img * composed_mask[..., np.newaxis]
        bboxes = updated_src_bboxes.cat([updated_dst_bboxes[valid_inds], updated_src_bboxes])
        labels = np.concatenate([dst_labels[valid_inds], src_labels])
        masks = np.concatenate([updated_dst_masks.masks[valid_inds], updated_src_masks.masks])
        ignore_flags = np.concatenate([dst_ignore_flags[valid_inds], src_ignore_flags])

        dst_results['img'] = img
        dst_results['gt_bboxes'] = bboxes
        dst_results['gt_bboxes_labels'] = labels
        dst_results['gt_masks'] = BitmapMasks(masks, masks.shape[1],
                                              masks.shape[2])
        dst_results['gt_ignore_flags'] = ignore_flags

        save_path_img = 'work_dirs/_img_copypaste/1/img'
        save_path_mask = 'work_dirs/_img_copypaste/1/mask'
        if not os.path.exists(save_path_img):
            os.makedirs(save_path_img)
        if not os.path.exists(save_path_mask):
            os.makedirs(save_path_mask)
        img_save_path = os.path.join(save_path_img, osp.basename(dst_results['img_path']))
        cv2.imwrite(img_save_path, dst_results['img'])
        gt_masks_save_path = os.path.join(save_path_mask, osp.basename(dst_results['img_path']))
        gt_masks = dst_results['gt_masks'].masks.astype('uint8') * 255
        merged_mask = np.any(gt_masks, axis=0).astype('uint8') * 255
        cv2.imwrite(gt_masks_save_path, merged_mask)

        return dst_results
    
    
    def _copy_paste_2(self, dst_results: dict, src_results: dict) -> dict:
        """CopyPaste transform function.

        Args:
            dst_results (dict): Result dict of the destination image.
            src_results (dict): Result dict of the source image.
        Returns:
            dict: Updated result dict.
        """
        dst_img = dst_results['img']
        dst_bboxes = dst_results['gt_bboxes']
        dst_labels = dst_results['gt_bboxes_labels']
        dst_masks = self.get_gt_masks(dst_results)
        dst_ignore_flags = dst_results['gt_ignore_flags']

        src_img = src_results['img']
        src_bboxes = src_results['gt_bboxes']
        src_labels = src_results['gt_bboxes_labels']
        src_masks = self.get_gt_masks(src_results)
        src_ignore_flags = src_results['gt_ignore_flags']

        if len(src_bboxes) == 0:
            return dst_results

        # generate new mask and bboxes
        updated_src_mask = np.zeros_like(src_masks.masks)

        # 读取原图像mask
        img_basename = osp.basename(dst_results['img_path'])
        dst_mask_path = osp.join("data/SIRSTdevkit/SIRST/BinaryMask", img_basename.replace('.png', '_pixels0.png'))
        mask_img = Image.open(dst_mask_path)
        dst_mask_gt = np.array(mask_img)
        dst_mask_gt = cv2.resize(dst_mask_gt, (512, 512), interpolation=cv2.INTER_NEAREST)
        if dst_results['flip']:
            dst_mask_gt = cv2.flip(dst_mask_gt, 1) # 水平翻转
        target_mask = dst_mask_gt

        offset = 25
        for i, src_bbox in enumerate(src_bboxes):
            x1, x2 = int(src_bbox.tensor[0, 0].item()), int(src_bbox.tensor[0, 2].item())
            y1, y2 = int(src_bbox.tensor[0, 1].item()), int(src_bbox.tensor[0, 3].item())
            width = x2 - x1
            height = y2 - y1

            if width >= 10 or height >= 10 :
                width = 7
                height = 7

            if i % 5 == 0:
                while True:
                    find, sky_x, sky_y = self.choose_sky(target_mask, offset)
                    if find and sky_x < sky_x+offset-height and sky_y < sky_y+offset-width:
                        break
                    else:
                        offset -= 3

            flag = 0
            # 随机选择位置进行粘贴
            while True:
                x = np.random.randint(sky_x, sky_x+offset-height)
                y = np.random.randint(sky_y, sky_y+offset-width)

                target_overlap_sky = target_mask[x-1:x+height+1, y-1:y+width+1].any()
                if not target_overlap_sky:
                    target_mask[x:x + height, y:y + width] = 255
                    updated_src_mask[i][x:x + height, y:y + width] = 1
                    break
                else:
                    if flag > 100:
                        x, y = self.choose_point(target_mask, height, width)
                        target_mask[x:x + height, y:y + width] = 255
                        updated_src_mask[i][x:x + height, y:y + width] = 1
                        break
                    else:
                        flag += 1
                        continue

        updated_src_masks = BitmapMasks(updated_src_mask, updated_src_mask.shape[1], updated_src_mask.shape[2])
        updated_src_bboxes = updated_src_masks.get_bboxes(type(src_bboxes))

        # generate new img
        gray_img = src_img[:, :, 0]
        new_src_img_ = gray_img.astype(np.float32)
        # new_src_img_ = np.zeros_like(gray_img, dtype=float)
        img_count_mask = np.zeros_like(gray_img, dtype=float)
        for (bbox, updated_bbox) in zip(src_bboxes, updated_src_bboxes):
            x1, x2 = int(bbox.tensor[0, 0].item()), int(bbox.tensor[0, 2].item())
            y1, y2 = int(bbox.tensor[0, 1].item()), int(bbox.tensor[0, 3].item())
            new_x1, new_x2 = int(updated_bbox.tensor[0, 0].item()), int(updated_bbox.tensor[0, 2].item())
            new_y1, new_y2 = int(updated_bbox.tensor[0, 1].item()), int(updated_bbox.tensor[0, 3].item())
            if new_src_img_[new_y1:new_y2, new_x1:new_x2].shape==(0,0):
                continue
            
            choose_src = gray_img[y1:y2, x1:x2]
            # 大目标缩小
            height = y2 - y1
            width = x2 - x1
            if height >= 10 or width >= 10 :
                choose_src = cv2.resize(choose_src, (7, 7))

            new_h = new_y2-new_y1
            new_w = new_x2-new_x1
            for i in range(new_h):
                for j in range(new_w):
                    new_src_img_[new_y1 + i, new_x1 + j] = choose_src[i, j]
            # new_src_img_[new_y1:new_y2, new_x1:new_x2] = choose_src
            img_count_mask[new_y1:new_y2, new_x1:new_x2] += 1
        img_non_zero_count_indices = img_count_mask > 0
        new_src_img_[img_non_zero_count_indices] /= img_count_mask[img_non_zero_count_indices]
        new_src_img = new_src_img_.astype(np.uint8)
        new_src_img = np.stack([new_src_img, new_src_img, new_src_img], axis=-1)

        # update masks and generate bboxes from updated masks
        composed_mask = np.where(np.any(updated_src_masks.masks, axis=0), 1, 0)
        updated_dst_masks = self._get_updated_masks(dst_masks, composed_mask)
        updated_dst_bboxes = updated_dst_masks.get_bboxes(type(dst_bboxes))
        assert len(updated_dst_bboxes) == len(updated_dst_masks)

        # filter totally occluded objects
        l1_distance = (updated_dst_bboxes.tensor - dst_bboxes.tensor).abs()
        bboxes_inds = (l1_distance <= self.bbox_occluded_thr).all(dim=-1).numpy()
        masks_inds = updated_dst_masks.masks.sum(axis=(1, 2)) > self.mask_occluded_thr
        valid_inds = bboxes_inds | masks_inds

        # Paste source objects to destination image directly
        img = dst_img * (1 - composed_mask[..., np.newaxis]) + new_src_img * composed_mask[..., np.newaxis]
        bboxes = updated_src_bboxes.cat([updated_dst_bboxes[valid_inds], updated_src_bboxes])
        labels = np.concatenate([dst_labels[valid_inds], src_labels])
        masks = np.concatenate([updated_dst_masks.masks[valid_inds], updated_src_masks.masks])
        ignore_flags = np.concatenate([dst_ignore_flags[valid_inds], src_ignore_flags])

        dst_results['img'] = img
        dst_results['gt_bboxes'] = bboxes
        dst_results['gt_bboxes_labels'] = labels
        dst_results['gt_masks'] = BitmapMasks(masks, masks.shape[1], masks.shape[2])
        dst_results['gt_ignore_flags'] = ignore_flags

        save_path_img = 'work_dirs/_img_copypaste/2/img'
        save_path_mask = 'work_dirs/_img_copypaste/2/mask'
        if not os.path.exists(save_path_img):
            os.makedirs(save_path_img)
        if not os.path.exists(save_path_mask):
            os.makedirs(save_path_mask)
        img_save_path = os.path.join(save_path_img, osp.basename(dst_results['img_path']))
        cv2.imwrite(img_save_path, dst_results['img'])
        gt_masks_save_path = os.path.join(save_path_mask, osp.basename(dst_results['img_path']))
        gt_masks = dst_results['gt_masks'].masks.astype('uint8') * 255
        merged_mask = np.any(gt_masks, axis=0).astype('uint8') * 255
        cv2.imwrite(gt_masks_save_path, merged_mask)

        return dst_results
    

    def _copy_paste_3(self, dst_results: dict, src_results: dict) -> dict:
        """CopyPaste transform function.

        Args:
            dst_results (dict): Result dict of the destination image.
            src_results (dict): Result dict of the source image.
        Returns:
            dict: Updated result dict.
        """
        dst_img = dst_results['img']
        dst_bboxes = dst_results['gt_bboxes']
        dst_labels = dst_results['gt_bboxes_labels']
        dst_masks = self.get_gt_masks(dst_results)
        dst_ignore_flags = dst_results['gt_ignore_flags']

        src_img = src_results['img']
        src_bboxes = src_results['gt_bboxes']
        src_labels = src_results['gt_bboxes_labels']
        src_masks = self.get_gt_masks(src_results)
        src_ignore_flags = src_results['gt_ignore_flags']

        if len(src_bboxes) == 0:
            return dst_results

        # generate new mask and bboxes
        updated_src_mask = np.zeros_like(src_masks.masks)

        # 读取原图像mask
        img_basename = osp.basename(dst_results['img_path'])
        dst_mask_path = osp.join("data/SIRSTdevkit/SIRST/BinaryMask", img_basename.replace('.png', '_pixels0.png'))
        mask_img = Image.open(dst_mask_path)
        dst_mask_gt = np.array(mask_img)
        dst_mask_gt = cv2.resize(dst_mask_gt, (512, 512), interpolation=cv2.INTER_NEAREST)
        if dst_results['flip']:
            dst_mask_gt = cv2.flip(dst_mask_gt, 1) # 水平翻转
        target_mask = dst_mask_gt

        for i, src_bbox in enumerate(src_bboxes):
            x1, x2 = int(src_bbox.tensor[0, 0].item()), int(src_bbox.tensor[0, 2].item())
            y1, y2 = int(src_bbox.tensor[0, 1].item()), int(src_bbox.tensor[0, 3].item())
            width = x2 - x1
            height = y2 - y1

            if width >= 10 or height >= 10 :
                width = 7
                height = 7

            # 随机选择位置进行粘贴
            while True:
                y = np.random.randint(0, target_mask.shape[1])
                x = np.random.randint(0, target_mask.shape[0])

                target_overlap_sky = target_mask[x-1:x+height+1, y-1:y+width+1].any()
                if not target_overlap_sky:
                    target_mask[x:x + height, y:y + width] = 255
                    updated_src_mask[i][x:x + height, y:y + width] = 1
                    break
                else:
                    continue

        updated_src_masks = BitmapMasks(updated_src_mask, updated_src_mask.shape[1], updated_src_mask.shape[2])
        updated_src_bboxes = updated_src_masks.get_bboxes(type(src_bboxes))

        # generate new img
        gray_img = src_img[:, :, 0]
        gray_img_dst = dst_img[:, :, 0]
        new_src_img_ = gray_img.astype(np.float32)
        # new_src_img_ = np.zeros_like(gray_img, dtype=float)
        img_count_mask = np.zeros_like(gray_img, dtype=float)
        for (bbox, updated_bbox) in zip(src_bboxes, updated_src_bboxes):
            x1, x2 = int(bbox.tensor[0, 0].item()), int(bbox.tensor[0, 2].item())
            y1, y2 = int(bbox.tensor[0, 1].item()), int(bbox.tensor[0, 3].item())
            new_x1, new_x2 = int(updated_bbox.tensor[0, 0].item()), int(updated_bbox.tensor[0, 2].item())
            new_y1, new_y2 = int(updated_bbox.tensor[0, 1].item()), int(updated_bbox.tensor[0, 3].item())
            if new_src_img_[new_y1:new_y2, new_x1:new_x2].shape==(0,0):
                continue
            
            choose_src = gray_img[y1:y2, x1:x2]
            # 大目标缩小
            height = y2 - y1
            width = x2 - x1
            if height >= 10 or width >= 10 :
                choose_src = cv2.resize(choose_src, (7, 7))

            # 高斯
            lamda = np.random.uniform(0.5, 1)
            new_h = new_y2-new_y1
            new_w = new_x2-new_x1
            gaussian_matrix = self.generate_gaussian_matrix(new_h, new_w)

            for i in range(new_h):
                for j in range(new_w):
                    pixel_value = gray_img_dst[new_y1 + i, new_x1 + j] + choose_src[i, j] * lamda * gaussian_matrix[i, j]
                    new_src_img_[new_y1 + i, new_x1 + j] = min(255, pixel_value)  # 确保值不超过255
            gauss_mask = np.ones_like(img_count_mask[new_y1:new_y2, new_x1:new_x2]) * 255
            gauss_mask = gauss_mask * lamda * gaussian_matrix
            ret, binary = cv2.threshold(gauss_mask, 25, 255, cv2.THRESH_BINARY)
            binary_height, binary_width = binary.shape
            for i in range(binary_height):
                for j in range(binary_width):
                    if binary[i, j] == 255:
                        img_count_mask[new_y1 + i, new_x1 + j] += 1

            # new_src_img_[new_y1:new_y2, new_x1:new_x2] = choose_src
            # img_count_mask[new_y1:new_y2, new_x1:new_x2] += 1
        img_non_zero_count_indices = img_count_mask > 0
        new_src_img_[img_non_zero_count_indices] /= img_count_mask[img_non_zero_count_indices]
        new_src_img = new_src_img_.astype(np.uint8)
        new_src_img = np.stack([new_src_img, new_src_img, new_src_img], axis=-1)

        # update masks and generate bboxes from updated masks
        composed_mask = np.where(np.any(updated_src_masks.masks, axis=0), 1, 0)
        updated_dst_masks = self._get_updated_masks(dst_masks, composed_mask)
        updated_dst_bboxes = updated_dst_masks.get_bboxes(type(dst_bboxes))
        assert len(updated_dst_bboxes) == len(updated_dst_masks)

        # filter totally occluded objects
        l1_distance = (updated_dst_bboxes.tensor - dst_bboxes.tensor).abs()
        bboxes_inds = (l1_distance <= self.bbox_occluded_thr).all(dim=-1).numpy()
        masks_inds = updated_dst_masks.masks.sum(axis=(1, 2)) > self.mask_occluded_thr
        valid_inds = bboxes_inds | masks_inds

        # Paste source objects to destination image directly
        img = dst_img * (1 - composed_mask[..., np.newaxis]) + new_src_img * composed_mask[..., np.newaxis]
        bboxes = updated_src_bboxes.cat([updated_dst_bboxes[valid_inds], updated_src_bboxes])
        labels = np.concatenate([dst_labels[valid_inds], src_labels])
        masks = np.concatenate([updated_dst_masks.masks[valid_inds], updated_src_masks.masks])
        ignore_flags = np.concatenate([dst_ignore_flags[valid_inds], src_ignore_flags])

        dst_results['img'] = img
        dst_results['gt_bboxes'] = bboxes
        dst_results['gt_bboxes_labels'] = labels
        dst_results['gt_masks'] = BitmapMasks(masks, masks.shape[1], masks.shape[2])
        dst_results['gt_ignore_flags'] = ignore_flags

        save_path_img = 'work_dirs/_img_copypaste/3/img'
        save_path_mask = 'work_dirs/_img_copypaste/3/mask'
        if not os.path.exists(save_path_img):
            os.makedirs(save_path_img)
        if not os.path.exists(save_path_mask):
            os.makedirs(save_path_mask)
        img_save_path = os.path.join(save_path_img, osp.basename(dst_results['img_path']))
        cv2.imwrite(img_save_path, dst_results['img'])
        gt_masks_save_path = os.path.join(save_path_mask, osp.basename(dst_results['img_path']))
        gt_masks = dst_results['gt_masks'].masks.astype('uint8') * 255
        merged_mask = np.any(gt_masks, axis=0).astype('uint8') * 255
        cv2.imwrite(gt_masks_save_path, merged_mask)

        return dst_results
    

    def _copy_paste_4(self, dst_results: dict, src_results: dict) -> dict:
        """CopyPaste transform function.

        Args:
            dst_results (dict): Result dict of the destination image.
            src_results (dict): Result dict of the source image.
        Returns:
            dict: Updated result dict.
        """
        dst_img = dst_results['img']
        dst_bboxes = dst_results['gt_bboxes']
        dst_labels = dst_results['gt_bboxes_labels']
        dst_masks = self.get_gt_masks(dst_results)
        dst_ignore_flags = dst_results['gt_ignore_flags']

        src_img = src_results['img']
        src_bboxes = src_results['gt_bboxes']
        src_labels = src_results['gt_bboxes_labels']
        src_masks = self.get_gt_masks(src_results)
        src_ignore_flags = src_results['gt_ignore_flags']

        if len(src_bboxes) == 0:
            return dst_results

        # generate new mask and bboxes
        updated_src_mask = np.zeros_like(src_masks.masks)

        # 读取原图像mask
        img_basename = osp.basename(dst_results['img_path'])
        dst_mask_path = osp.join("data/SIRSTdevkit/SIRST/BinaryMask", img_basename.replace('.png', '_pixels0.png'))
        mask_img = Image.open(dst_mask_path)
        dst_mask_gt = np.array(mask_img)
        dst_mask_gt = cv2.resize(dst_mask_gt, (512, 512), interpolation=cv2.INTER_NEAREST)
        if dst_results['flip']:
            dst_mask_gt = cv2.flip(dst_mask_gt, 1) # 水平翻转
        target_mask = dst_mask_gt

        offset = 25
        for i, src_bbox in enumerate(src_bboxes):
            x1, x2 = int(src_bbox.tensor[0, 0].item()), int(src_bbox.tensor[0, 2].item())
            y1, y2 = int(src_bbox.tensor[0, 1].item()), int(src_bbox.tensor[0, 3].item())
            width = x2 - x1
            height = y2 - y1

            if width >= 10 or height >= 10 :
                width = 7
                height = 7

            if i % 5 == 0:
                while True:
                    find, sky_x, sky_y = self.choose_sky(target_mask, offset)
                    if find and sky_x < sky_x+offset-height and sky_y < sky_y+offset-width:
                        break
                    else:
                        offset -= 3

            flag = 0
            # 随机选择位置进行粘贴
            while True:
                x = np.random.randint(sky_x, sky_x+offset-height)
                y = np.random.randint(sky_y, sky_y+offset-width)

                target_overlap_sky = target_mask[x-1:x+height+1, y-1:y+width+1].any()
                if not target_overlap_sky:
                    target_mask[x:x + height, y:y + width] = 255
                    updated_src_mask[i][x:x + height, y:y + width] = 1
                    break
                else:
                    if flag > 100:
                        x, y = self.choose_point(target_mask, height, width)
                        target_mask[x:x + height, y:y + width] = 255
                        updated_src_mask[i][x:x + height, y:y + width] = 1
                        break
                    else:
                        flag += 1
                        continue

        updated_src_masks = BitmapMasks(updated_src_mask, updated_src_mask.shape[1], updated_src_mask.shape[2])
        updated_src_bboxes = updated_src_masks.get_bboxes(type(src_bboxes))

        # generate new img
        gray_img = src_img[:, :, 0]
        gray_img_dst = dst_img[:, :, 0]
        new_src_img_ = gray_img.astype(np.float32)
        # new_src_img_ = np.zeros_like(gray_img, dtype=float)
        img_count_mask = np.zeros_like(gray_img, dtype=float)
        for (bbox, updated_bbox) in zip(src_bboxes, updated_src_bboxes):
            x1, x2 = int(bbox.tensor[0, 0].item()), int(bbox.tensor[0, 2].item())
            y1, y2 = int(bbox.tensor[0, 1].item()), int(bbox.tensor[0, 3].item())
            new_x1, new_x2 = int(updated_bbox.tensor[0, 0].item()), int(updated_bbox.tensor[0, 2].item())
            new_y1, new_y2 = int(updated_bbox.tensor[0, 1].item()), int(updated_bbox.tensor[0, 3].item())
            if new_src_img_[new_y1:new_y2, new_x1:new_x2].shape==(0,0):
                continue
            
            choose_src = gray_img[y1:y2, x1:x2]
            # 大目标缩小
            height = y2 - y1
            width = x2 - x1
            if height >= 10 or width >= 10 :
                choose_src = cv2.resize(choose_src, (7, 7))

            # 高斯
            lamda = np.random.uniform(0.5, 1)
            new_h = new_y2-new_y1
            new_w = new_x2-new_x1
            gaussian_matrix = self.generate_gaussian_matrix(new_h, new_w)

            for i in range(new_h):
                for j in range(new_w):
                    pixel_value = gray_img_dst[new_y1 + i, new_x1 + j] + choose_src[i, j] * lamda * gaussian_matrix[i, j]
                    new_src_img_[new_y1 + i, new_x1 + j] = min(255, pixel_value)  # 确保值不超过255
            gauss_mask = np.ones_like(img_count_mask[new_y1:new_y2, new_x1:new_x2]) * 255
            gauss_mask = gauss_mask * lamda * gaussian_matrix
            ret, binary = cv2.threshold(gauss_mask, 25, 255, cv2.THRESH_BINARY)
            binary_height, binary_width = binary.shape
            for i in range(binary_height):
                for j in range(binary_width):
                    if binary[i, j] == 255:
                        img_count_mask[new_y1 + i, new_x1 + j] += 1

            # new_src_img_[new_y1:new_y2, new_x1:new_x2] = choose_src
            # img_count_mask[new_y1:new_y2, new_x1:new_x2] += 1
        img_non_zero_count_indices = img_count_mask > 0
        new_src_img_[img_non_zero_count_indices] /= img_count_mask[img_non_zero_count_indices]
        new_src_img = new_src_img_.astype(np.uint8)
        new_src_img = np.stack([new_src_img, new_src_img, new_src_img], axis=-1)

        # update masks and generate bboxes from updated masks
        composed_mask = np.where(np.any(updated_src_masks.masks, axis=0), 1, 0)
        updated_dst_masks = self._get_updated_masks(dst_masks, composed_mask)
        updated_dst_bboxes = updated_dst_masks.get_bboxes(type(dst_bboxes))
        assert len(updated_dst_bboxes) == len(updated_dst_masks)

        # filter totally occluded objects
        l1_distance = (updated_dst_bboxes.tensor - dst_bboxes.tensor).abs()
        bboxes_inds = (l1_distance <= self.bbox_occluded_thr).all(dim=-1).numpy()
        masks_inds = updated_dst_masks.masks.sum(axis=(1, 2)) > self.mask_occluded_thr
        valid_inds = bboxes_inds | masks_inds

        # Paste source objects to destination image directly
        img = dst_img * (1 - composed_mask[..., np.newaxis]) + new_src_img * composed_mask[..., np.newaxis]
        bboxes = updated_src_bboxes.cat([updated_dst_bboxes[valid_inds], updated_src_bboxes])
        labels = np.concatenate([dst_labels[valid_inds], src_labels])
        masks = np.concatenate([updated_dst_masks.masks[valid_inds], updated_src_masks.masks])
        ignore_flags = np.concatenate([dst_ignore_flags[valid_inds], src_ignore_flags])

        dst_results['img'] = img
        dst_results['gt_bboxes'] = bboxes
        dst_results['gt_bboxes_labels'] = labels
        dst_results['gt_masks'] = BitmapMasks(masks, masks.shape[1], masks.shape[2])
        dst_results['gt_ignore_flags'] = ignore_flags

        save_path_img = 'work_dirs/_img_copypaste/4/img'
        save_path_mask = 'work_dirs/_img_copypaste/4/mask'
        if not os.path.exists(save_path_img):
            os.makedirs(save_path_img)
        if not os.path.exists(save_path_mask):
            os.makedirs(save_path_mask)
        img_save_path = os.path.join(save_path_img, osp.basename(dst_results['img_path']))
        cv2.imwrite(img_save_path, dst_results['img'])
        gt_masks_save_path = os.path.join(save_path_mask, osp.basename(dst_results['img_path']))
        gt_masks = dst_results['gt_masks'].masks.astype('uint8') * 255
        merged_mask = np.any(gt_masks, axis=0).astype('uint8') * 255
        cv2.imwrite(gt_masks_save_path, merged_mask)

        return dst_results


    def _get_updated_masks(self, masks: BitmapMasks,
                           composed_mask: np.ndarray) -> BitmapMasks:
        """Update masks with composed mask."""
        assert masks.masks.shape[-2:] == composed_mask.shape[-2:], \
            'Cannot compare two arrays of different size'
        masks.masks = np.where(composed_mask, 0, masks.masks)
        return masks

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(max_num_pasted={self.max_num_pasted}, '
        repr_str += f'bbox_occluded_thr={self.bbox_occluded_thr}, '
        repr_str += f'mask_occluded_thr={self.mask_occluded_thr}, '
        repr_str += f'selected={self.selected}), '
        repr_str += f'paste_by_box={self.paste_by_box})'
        return repr_str
    
    def generate_gaussian_matrix(self, h, w, sigma_x=0.8, sigma_y=0.6, mu_x=0.0, mu_y=0.0, theta=0.0):
        """
        Generate a 2D rotated anisotropic Gaussian matrix.
        """

        sigma_x = np.random.uniform(0.3, 0.6)
        sigma_y = np.random.uniform(0.3, 0.6)
        mu_x = -np.random.uniform(0, 0.2)
        mu_y = -np.random.uniform(0, 0.2)
        theta = np.random.randint(-90, 90)

        # Angle of rotation in radians
        theta_radians = np.radians(theta)

        # Rotation matrix
        rotation_matrix = np.array([[np.cos(theta_radians), -np.sin(theta_radians)],
                                    [np.sin(theta_radians), np.cos(theta_radians)]])

        # Create a coordinate grid
        X, Y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
        
        # Stack the coordinates for matrix multiplication
        coords = np.stack([X.ravel() - mu_x, Y.ravel() - mu_y])
        
        # Apply the rotation matrix to the coordinates
        rot_coords = rotation_matrix @ coords
        
        # Calculate the squared distance from the center for each axis after rotation
        d_x2 = (rot_coords[0] ** 2)
        d_y2 = (rot_coords[1] ** 2)
        
        # Apply the anisotropic Gaussian formula
        gaussian = np.exp(-(d_x2 / (2.0 * sigma_x**2) + d_y2 / (2.0 * sigma_y**2)))
        
        if gaussian.max() == 0:
            return gaussian
        else:
            # Reshape back to the original shape and normalize
            gaussian = gaussian.reshape(h, w)
            gaussian -= gaussian.min()
            gaussian /= gaussian.max()
            return gaussian
        
    def choose_sky(self, target_mask, offset):
        flag = 0
        find = True
        while True:
            # 随机选择一个区域粘贴簇
            y = np.random.randint(0, target_mask.shape[1])
            x = np.random.randint(0, target_mask.shape[0])

            overlap = target_mask[x:x+offset, y:y+offset].any()
            if not overlap:
                break
            else:
                if flag > 100:
                    find = False
                    break
                else:
                    flag += 1
                    continue

        return find, x, y
    

    def choose_point(self, target_mask, h, w):
        flag = 0
        while True:
            # 随机选择一个天空位置
            y = np.random.randint(0, target_mask.shape[1])
            x = np.random.randint(0, target_mask.shape[0])

            overlap = target_mask[x-1:x+h+1, y-1:y+w+1].any()
            if not overlap:
                break
            else:
                if flag > 100:
                    return x, y
                else:
                    flag += 1
                    continue

        return x, y


@TRANSFORMS.register_module()
class SkyCopyPaste(BaseTransform):
    """Simple Copy-Paste is a Strong Data Augmentation Method for Instance
    Segmentation The simple copy-paste transform steps are as follows:

    1. The destination image is already resized with aspect ratio kept,
       cropped and padded.
    2. Randomly select a source image, which is also already resized
       with aspect ratio kept, cropped and padded in a similar way
       as the destination image.
    3. Randomly select some objects from the source image.
    4. Paste these source objects to the destination image directly,
       due to the source and destination image have the same size.
    5. Update object masks of the destination image, for some origin objects
       may be occluded.
    6. Generate bboxes from the updated destination masks and
       filter some objects which are totally occluded, and adjust bboxes
       which are partly occluded.
    7. Append selected source bboxes, masks, and labels.

    Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_bboxes_labels (np.int64) (optional)
    - gt_ignore_flags (bool) (optional)
    - gt_masks (BitmapMasks) (optional)

    Modified Keys:

    - img
    - gt_bboxes (optional)
    - gt_bboxes_labels (optional)
    - gt_ignore_flags (optional)
    - gt_masks (optional)

    Args:
        max_num_pasted (int): The maximum number of pasted objects.
            Defaults to 100.
        bbox_occluded_thr (int): The threshold of occluded bbox.
            Defaults to 10.
        mask_occluded_thr (int): The threshold of occluded mask.
            Defaults to 300.
        selected (bool): Whether select objects or not. If select is False,
            all objects of the source image will be pasted to the
            destination image.
            Defaults to True.
        paste_by_box (bool): Whether use boxes as masks when masks are not
            available.
            Defaults to False.
    """

    def __init__(
            self,
            max_num_pasted: int = 100,
            bbox_occluded_thr: int = 10,
            mask_occluded_thr: int = 300,
            selected: bool = True,
            paste_by_box: bool = False,
    ) -> None:
        self.max_num_pasted = max_num_pasted
        self.bbox_occluded_thr = bbox_occluded_thr
        self.mask_occluded_thr = mask_occluded_thr
        self.selected = selected
        self.paste_by_box = paste_by_box

    @cache_randomness
    def get_indexes(self, dataset: BaseDataset) -> int:
        """Call function to collect indexes.s.

        Args:
            dataset (:obj:`MultiImageMixDataset`): The dataset.
        Returns:
            list: Indexes.
        """
        return random.randint(0, len(dataset))

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        """Transform function to make a copy-paste of image.

        Args:
            results (dict): Result dict.
        Returns:
            dict: Result dict with copy-paste transformed.
        """

        assert 'mix_results' in results
        num_images = len(results['mix_results'])
        assert num_images == 1, \
            f'CopyPaste only supports processing 2 images, got {num_images}'
        if self.selected:
            selected_results = self._select_object(results['mix_results'][0])
        else:
            selected_results = results['mix_results'][0]
        return self._copy_paste(results, selected_results)

    @cache_randomness
    def _get_selected_inds(self, num_bboxes: int) -> np.ndarray:
        max_num_pasted = min(num_bboxes + 1, self.max_num_pasted)
        num_pasted = np.random.randint(0, max_num_pasted)
        return np.random.choice(num_bboxes, size=num_pasted, replace=False)

    def get_gt_masks(self, results: dict) -> BitmapMasks:
        """Get gt_masks originally or generated based on bboxes.

        If gt_masks is not contained in results,
        it will be generated based on gt_bboxes.
        Args:
            results (dict): Result dict.
        Returns:
            BitmapMasks: gt_masks, originally or generated based on bboxes.
        """
        if results.get('gt_masks', None) is not None:
            if self.paste_by_box:
                warnings.warn('gt_masks is already contained in results, '
                              'so paste_by_box is disabled.')
            return results['gt_masks']
        else:
            if not self.paste_by_box:
                raise RuntimeError('results does not contain masks.')
            return results['gt_bboxes'].create_masks(results['img'].shape[:2])

    def _select_object(self, results: dict) -> dict:
        """Select some objects from the source results."""
        bboxes = results['gt_bboxes']
        labels = results['gt_bboxes_labels']
        masks = self.get_gt_masks(results)
        ignore_flags = results['gt_ignore_flags']

        selected_inds = self._get_selected_inds(bboxes.shape[0])

        selected_bboxes = bboxes[selected_inds]
        selected_labels = labels[selected_inds]
        selected_masks = masks[selected_inds]
        selected_ignore_flags = ignore_flags[selected_inds]

        results['gt_bboxes'] = selected_bboxes
        results['gt_bboxes_labels'] = selected_labels
        results['gt_masks'] = selected_masks
        results['gt_ignore_flags'] = selected_ignore_flags
        return results

    def _copy_paste(self, dst_results: dict, src_results: dict) -> dict:
        """CopyPaste transform function.

        Args:
            dst_results (dict): Result dict of the destination image.
            src_results (dict): Result dict of the source image.
        Returns:
            dict: Updated result dict.
        """
        dst_img = dst_results['img']
        dst_bboxes = dst_results['gt_bboxes']
        dst_labels = dst_results['gt_bboxes_labels']
        dst_masks = self.get_gt_masks(dst_results)
        dst_ignore_flags = dst_results['gt_ignore_flags']

        src_img = src_results['img']
        src_bboxes = src_results['gt_bboxes']
        src_labels = src_results['gt_bboxes_labels']
        src_masks = self.get_gt_masks(src_results)
        src_ignore_flags = src_results['gt_ignore_flags']

        if len(src_bboxes) == 0:
            return dst_results

        # sky mask
        img_basename = osp.basename(dst_results['img_path'])
        dst_sky_path = osp.join("data/SIRSTdevkit/SkySeg/BinaryMask", img_basename.replace('.png', '_pixels0.png'))
        dst_sky_gt = cv2.imread(dst_sky_path, cv2.IMREAD_GRAYSCALE)
        dst_sky_gt = cv2.resize(dst_sky_gt, (512, 512), interpolation=cv2.INTER_NEAREST)
        if dst_results['flip']:
            dst_sky_gt = cv2.flip(dst_sky_gt, 1) # 水平翻转
        assert dst_sky_gt.shape == dst_masks.masks.shape[-2:]
        
        # generate new mask and bboxes
        updated_src_mask = np.zeros_like(src_masks.masks)
        sky_coordinates = np.where(dst_sky_gt == 255)

        if len(sky_coordinates[0]) <= 0:
            return dst_results

        # 读取原图像mask
        dst_mask_path = osp.join("data/SIRSTdevkit/SIRST/BinaryMask", img_basename.replace('.png', '_pixels0.png'))
        dst_mask_gt = cv2.imread(dst_mask_path, cv2.IMREAD_GRAYSCALE)
        dst_mask_gt = cv2.resize(dst_mask_gt, (512, 512), interpolation=cv2.INTER_NEAREST)
        if dst_results['flip']:
            dst_mask_gt = cv2.flip(dst_mask_gt, 1) # 水平翻转
        target_mask = dst_mask_gt

        offset = 25
        for i, src_bbox in enumerate(src_bboxes):
            x1, x2 = int(src_bbox.tensor[0, 0].item()), int(src_bbox.tensor[0, 2].item())
            y1, y2 = int(src_bbox.tensor[0, 1].item()), int(src_bbox.tensor[0, 3].item())
            width = x2 - x1
            height = y2 - y1

            if width >= 10 or height >= 10 :
                width = 7
                height = 7

            if i % 5 == 0:
                flag_find = True
                count_find = 0
                while True:
                    find, sky_x, sky_y = self.choose_sky(dst_sky_gt, target_mask, offset)
                    if find and sky_x < sky_x+offset-7 and sky_y < sky_y+offset-7:
                        break
                    else:
                        if count_find > 5:
                            offset = 25
                            flag_find = False
                            break
                        else:
                            offset -= 3
                            count_find += 1
                            

            if not flag_find:
                x, y = self.choose_point(dst_sky_gt, target_mask, height, width)
                target_mask[x:x + height, y:y + width] = 255
                updated_src_mask[i][x:x + height, y:y + width] = 1
            else:
                flag = 0
                # 随机选择天空位置进行粘贴
                while True:
                    x = np.random.randint(sky_x, sky_x+offset-height)
                    y = np.random.randint(sky_y, sky_y+offset-width)

                    target_overlap_sky = target_mask[x-1:x+height+1, y-1:y+width+1].any()

                    # 判断选取的区域是否完全是天空区域
                    random_sky = np.zeros_like(dst_sky_gt)
                    random_sky[x:x+height, y:y+width] = 255
                    # 使用位运算检查随机区域是否完全在天空区域内
                    random_sky_vs = cv2.bitwise_and(random_sky, dst_sky_gt)
                    is_completely_in_sky = np.all(random_sky_vs == random_sky)
                    
                    if is_completely_in_sky and not target_overlap_sky:
                        target_mask[x:x + height, y:y + width] = 255
                        updated_src_mask[i][x:x + height, y:y + width] = 1
                        break
                    else:
                        if flag > 100:
                            x, y = self.choose_point(dst_sky_gt, target_mask, height, width)
                            target_mask[x:x + height, y:y + width] = 255
                            updated_src_mask[i][x:x + height, y:y + width] = 1
                            break
                        else:
                            flag += 1
                            continue

        updated_src_masks = BitmapMasks(updated_src_mask, updated_src_mask.shape[1], updated_src_mask.shape[2])
        updated_src_bboxes = updated_src_masks.get_bboxes(type(src_bboxes))

        # generate new img
        gray_img = src_img[:, :, 0]
        gray_img_dst = dst_img[:, :, 0]
        new_src_img_ = gray_img.astype(np.float32)
        # new_src_img_ = np.zeros_like(gray_img, dtype=float)
        img_count_mask = np.zeros_like(gray_img, dtype=float)
        for (bbox, updated_bbox) in zip(src_bboxes, updated_src_bboxes):
            x1, x2 = int(bbox.tensor[0, 0].item()), int(bbox.tensor[0, 2].item())
            y1, y2 = int(bbox.tensor[0, 1].item()), int(bbox.tensor[0, 3].item())
            new_x1, new_x2 = int(updated_bbox.tensor[0, 0].item()), int(updated_bbox.tensor[0, 2].item())
            new_y1, new_y2 = int(updated_bbox.tensor[0, 1].item()), int(updated_bbox.tensor[0, 3].item())
            if new_src_img_[new_y1:new_y2, new_x1:new_x2].shape==(0,0):
                continue
            
            choose_src = gray_img[y1:y2, x1:x2]
            # 大目标缩小
            height = y2 - y1
            width = x2 - x1
            if height >= 10 or width >= 10 :
                choose_src = cv2.resize(choose_src, (7, 7))

            # 高斯
            lamda = np.random.uniform(0.5, 1)
            new_h = new_y2-new_y1
            new_w = new_x2-new_x1
            gaussian_matrix = self.generate_gaussian_matrix(new_h, new_w)

            for i in range(new_h):
                for j in range(new_w):
                    pixel_value = gray_img_dst[new_y1 + i, new_x1 + j] + choose_src[i, j] * lamda * gaussian_matrix[i, j]
                    new_src_img_[new_y1 + i, new_x1 + j] = min(255, pixel_value)  # 确保值不超过255
            gauss_mask = np.ones_like(img_count_mask[new_y1:new_y2, new_x1:new_x2]) * 255
            gauss_mask = gauss_mask * lamda * gaussian_matrix
            ret, binary = cv2.threshold(gauss_mask, 25, 255, cv2.THRESH_BINARY)
            binary_height, binary_width = binary.shape
            for i in range(binary_height):
                for j in range(binary_width):
                    if binary[i, j] == 255:
                        img_count_mask[new_y1 + i, new_x1 + j] += 1

            # new_src_img_[new_y1:new_y2, new_x1:new_x2] = gray_img[y1:y2, x1:x2]
            # img_count_mask[new_y1:new_y2, new_x1:new_x2] += 1
        img_non_zero_count_indices = img_count_mask > 0
        new_src_img_[img_non_zero_count_indices] /= img_count_mask[img_non_zero_count_indices]
        new_src_img = new_src_img_.astype(np.uint8)
        new_src_img = np.stack([new_src_img, new_src_img, new_src_img], axis=-1)

        # update masks and generate bboxes from updated masks
        composed_mask = np.where(np.any(updated_src_masks.masks, axis=0), 1, 0)
        updated_dst_masks = self._get_updated_masks(dst_masks, composed_mask)
        updated_dst_bboxes = updated_dst_masks.get_bboxes(type(dst_bboxes))
        assert len(updated_dst_bboxes) == len(updated_dst_masks)

        # filter totally occluded objects
        l1_distance = (updated_dst_bboxes.tensor - dst_bboxes.tensor).abs()
        bboxes_inds = (l1_distance <= self.bbox_occluded_thr).all(dim=-1).numpy()
        masks_inds = updated_dst_masks.masks.sum(axis=(1, 2)) > self.mask_occluded_thr
        valid_inds = bboxes_inds | masks_inds

        # Paste source objects to destination image directly
        img = dst_img * (1 - composed_mask[..., np.newaxis]) + new_src_img * composed_mask[..., np.newaxis]
        bboxes = updated_src_bboxes.cat([updated_dst_bboxes[valid_inds], updated_src_bboxes])
        labels = np.concatenate([dst_labels[valid_inds], src_labels])
        masks = np.concatenate([updated_dst_masks.masks[valid_inds], updated_src_masks.masks])
        ignore_flags = np.concatenate([dst_ignore_flags[valid_inds], src_ignore_flags])

        dst_results['img'] = img
        dst_results['gt_bboxes'] = bboxes
        dst_results['gt_bboxes_labels'] = labels
        dst_results['gt_masks'] = BitmapMasks(masks, masks.shape[1], masks.shape[2])
        dst_results['gt_ignore_flags'] = ignore_flags

        # save_path_img = 'work_dirs/_img_copypaste/5/img'
        # save_path_mask = 'work_dirs/_img_copypaste/5/mask'
        # if not os.path.exists(save_path_img):
        #     os.makedirs(save_path_img)
        # if not os.path.exists(save_path_mask):
        #     os.makedirs(save_path_mask)
        # img_save_path = os.path.join(save_path_img, osp.basename(dst_results['img_path']))
        # cv2.imwrite(img_save_path, dst_results['img'])
        # gt_masks_save_path = os.path.join(save_path_mask, osp.basename(dst_results['img_path']))
        # gt_masks = dst_results['gt_masks'].masks.astype('uint8') * 255
        # merged_mask = np.any(gt_masks, axis=0).astype('uint8') * 255
        # cv2.imwrite(gt_masks_save_path, merged_mask)

        return dst_results


    def _get_updated_masks(self, masks: BitmapMasks,
                           composed_mask: np.ndarray) -> BitmapMasks:
        """Update masks with composed mask."""
        assert masks.masks.shape[-2:] == composed_mask.shape[-2:], \
            'Cannot compare two arrays of different size'
        masks.masks = np.where(composed_mask, 0, masks.masks)
        return masks

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(max_num_pasted={self.max_num_pasted}, '
        repr_str += f'bbox_occluded_thr={self.bbox_occluded_thr}, '
        repr_str += f'mask_occluded_thr={self.mask_occluded_thr}, '
        repr_str += f'selected={self.selected}), '
        repr_str += f'paste_by_box={self.paste_by_box})'
        return repr_str
    
    def generate_gaussian_matrix(self, h, w, sigma_x=0.8, sigma_y=0.6, mu_x=0.0, mu_y=0.0, theta=0.0):
        """
        Generate a 2D rotated anisotropic Gaussian matrix.
        """

        sigma_x = np.random.uniform(0.3, 0.6)
        sigma_y = np.random.uniform(0.3, 0.6)
        mu_x = -np.random.uniform(0, 0.2)
        mu_y = -np.random.uniform(0, 0.2)
        theta = np.random.randint(-90, 90)

        # Angle of rotation in radians
        theta_radians = np.radians(theta)

        # Rotation matrix
        rotation_matrix = np.array([[np.cos(theta_radians), -np.sin(theta_radians)],
                                    [np.sin(theta_radians), np.cos(theta_radians)]])

        # Create a coordinate grid
        X, Y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
        
        # Stack the coordinates for matrix multiplication
        coords = np.stack([X.ravel() - mu_x, Y.ravel() - mu_y])
        
        # Apply the rotation matrix to the coordinates
        rot_coords = rotation_matrix @ coords
        
        # Calculate the squared distance from the center for each axis after rotation
        d_x2 = (rot_coords[0] ** 2)
        d_y2 = (rot_coords[1] ** 2)
        
        # Apply the anisotropic Gaussian formula
        gaussian = np.exp(-(d_x2 / (2.0 * sigma_x**2) + d_y2 / (2.0 * sigma_y**2)))
        
        if gaussian.max() == 0:
            return gaussian
        else:
            # Reshape back to the original shape and normalize
            gaussian = gaussian.reshape(h, w)
            gaussian -= gaussian.min()
            gaussian /= gaussian.max()
            return gaussian
        
    def choose_sky(self, dst_sky_gt, target_mask, offset):
        sky_coordinates = np.where(dst_sky_gt == 255)
        flag = 0
        find = True
        while True:
            # 随机选择一个天空区域粘贴簇
            random_sky = np.random.randint(0, len(sky_coordinates[0]))
            y = min(dst_sky_gt.shape[1] - offset, sky_coordinates[1][random_sky]) # 列坐标
            x = min(dst_sky_gt.shape[0] - offset, sky_coordinates[0][random_sky]) # 行坐标
            overlap = target_mask[x:x+offset, y:y+offset].any()
            # 判断选取的区域是否完全是天空区域
            random_sky_offset = np.zeros_like(dst_sky_gt)
            random_sky_offset[x:x+offset, y:y+offset] = 255
            # 使用位运算检查随机区域是否完全在天空区域内
            random_vs = cv2.bitwise_and(random_sky_offset, dst_sky_gt)
            is_completely_in_sky = np.all(random_vs == random_sky_offset)
            
            if is_completely_in_sky and not overlap:
                break
            else:
                if flag > 100:
                    find = False
                    break
                else:
                    flag += 1
                    continue

        return find, x, y
    

    def choose_point(self, dst_sky_gt, target_mask, h, w):
        sky_coordinates = np.where(dst_sky_gt == 255)
        flag = 0
        while True:
            # 随机选择一个天空位置
            random_sky = np.random.randint(0, len(sky_coordinates[0]))
            y = min(dst_sky_gt.shape[1] - 1, sky_coordinates[1][random_sky]) # 列坐标
            x = min(dst_sky_gt.shape[0] - 1, sky_coordinates[0][random_sky]) # 行坐标
            overlap = target_mask[x-1:x+h+1, y-1:y+w+1].any()
            # 判断选取的区域是否完全是天空区域
            random_sky_offset = np.zeros_like(dst_sky_gt)
            random_sky_offset[x:x+h, y:y+w] = 255
            # 使用位运算检查随机区域是否完全在天空区域内
            random_vs = cv2.bitwise_and(random_sky_offset, dst_sky_gt)
            is_completely_in_sky = np.all(random_vs == random_sky_offset)
            
            if is_completely_in_sky and not overlap:
                break
            else:
                if flag > 100:
                    return x, y
                else:
                    flag += 1
                    continue

        return x, y