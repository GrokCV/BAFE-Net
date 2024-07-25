# Copyright (c) GrokCV. All rights reserved.
import copy
import os.path as osp
from typing import Callable, Dict, List, Optional, Sequence, Union

import mmengine
import mmengine.fileio as fileio

from mmseg.datasets import BaseSegDataset

from deepir.registry import DATASETS


@DATASETS.register_module()
class SIRSTDualSegDataset(BaseSegDataset):
    """SIRST Segmentation Dataset in Pascal VOC format.

    Args:
        split (str): Split txt file for Pascal VOC.
    """
    METAINFO = dict(
        classes=('background', 'target'),
        palette=[[0, 0, 0], [128, 0, 0]],
        bg_classes=('others', 'sky'),
        bg_palette=[[0, 0, 0], [128, 0, 0]]
        )

    def __init__(self,
                 ann_file,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 bg_map_suffix='.png',
                 data_prefix: dict = dict(img_path='PNGImages', seg_map_path='SIRST/BinaryMask', bg_map_path='SkySeg/BinaryMask'),
                 **kwargs) -> None:
        # 必须把 bg_map_suffix 赋值放到 super 之前
        # 否则会报错找不到 bg_map_suffix
        self.bg_map_suffix = bg_map_suffix
        super().__init__(
            ann_file=ann_file,
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            data_prefix=data_prefix,
            **kwargs)
        assert fileio.exists(self.data_prefix['img_path'],
                             self.backend_args) and osp.isfile(self.ann_file)

    def load_data_list(self) -> List[dict]:
        """Load annotation from directory or annotation file.

        Returns:
            list[dict]: All data info of dataset.
        """
        data_list = []
        img_dir = self.data_prefix.get('img_path', None)
        ann_dir = self.data_prefix.get('seg_map_path', None)
        bg_dir = self.data_prefix.get('bg_map_path', None)
        if osp.isfile(self.ann_file):
            lines = mmengine.list_from_file(
                self.ann_file, backend_args=self.backend_args)
            for line in lines:
                img_name = line.strip()
                data_info = dict(
                    img_path=osp.join(img_dir, img_name + self.img_suffix))
                if ann_dir is not None:
                    seg_map = img_name + self.seg_map_suffix
                    data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
                if bg_dir is not None:
                    bg_map = img_name + self.bg_map_suffix
                    data_info['bg_map_path'] = osp.join(bg_dir, bg_map)
                data_info['label_map'] = self.label_map
                data_info['reduce_zero_label'] = self.reduce_zero_label
                data_info['seg_fields'] = []
                data_list.append(data_info)
        else:
            for img in fileio.list_dir_or_file(
                    dir_path=img_dir,
                    list_dir=False,
                    suffix=self.img_suffix,
                    recursive=True,
                    backend_args=self.backend_args):
                data_info = dict(img_path=osp.join(img_dir, img))
                if ann_dir is not None:
                    seg_map = img.replace(self.img_suffix, self.seg_map_suffix)
                    data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
                if bg_dir is not None:
                    bg_map = img.replace(self.img_suffix, self.bg_map_suffix)
                    data_info['bg_map_path'] = osp.join(bg_dir, bg_map)
                data_info['label_map'] = self.label_map
                data_info['reduce_zero_label'] = self.reduce_zero_label
                data_info['seg_fields'] = []
                data_list.append(data_info)
            data_list = sorted(data_list, key=lambda x: x['img_path'])
        return data_list