# Copyright (c) GrokCV. All rights reserved.
import os.path as osp
import mmengine.fileio as fileio
from typing import List
from mmseg.datasets import BaseSegDataset
import mmengine
from deepir.registry import DATASETS


@DATASETS.register_module()
class SIRSTSegDataset(BaseSegDataset):
    """SIRST Segmentation Dataset in Pascal VOC format.

    Args:
        split (str): Split txt file for Pascal VOC.
    """
    METAINFO = dict(
        classes=('background', 'target'),
        palette=[[0, 0, 0], [255, 0, 0]])
    
    # METAINFO = {
    #     "classes": ("Target",),
    #     "palette": [
    #         (255, 0, 0),
    #     ],
    # }

    def __init__(self,
                 ann_file,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 ignore_index=255,
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            ann_file=ann_file,
            ignore_index=ignore_index,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
        # print("self.ann_file:", self.ann_file)
        # print("self.data_prefix:", self.data_prefix)
        # print("self.data_prefix['img_path']:", self.data_prefix['img_path'])
        # print("fileio.exists(self.data_prefix['img_path'], self.backend_args):", fileio.exists(self.data_prefix['img_path'], self.backend_args))
        # print("osp.isfile(self.ann_file):", osp.isfile(self.ann_file))
        assert fileio.exists(self.data_prefix['img_path'],
                             self.backend_args) and osp.isfile(self.ann_file)
        
    # def load_data_list(self) -> List[dict]:
    #     """Load annotation from directory or annotation file.

    #     Returns:
    #         list[dict]: All data info of dataset.
    #     """
    #     data_list = []
    #     img_dir = self.data_prefix.get('img_path', None)
    #     ann_dir = self.data_prefix.get('seg_map_path', None)
    #     if osp.isfile(self.ann_file):
    #         lines = mmengine.list_from_file(
    #             self.ann_file, backend_args=self.backend_args)
    #         for line in lines:
    #             img_name = line.strip()
    #             data_info = dict(
    #                 img_path=osp.join(img_dir, img_name + self.img_suffix))
    #             if ann_dir is not None:
    #                 seg_map = img_name + "_pixels0" + self.seg_map_suffix
    #                 data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
    #             data_info['label_map'] = self.label_map
    #             data_info['reduce_zero_label'] = self.reduce_zero_label
    #             data_info['seg_fields'] = []
    #             data_list.append(data_info)
    #     else:
    #         for img in fileio.list_dir_or_file(
    #                 dir_path=img_dir,
    #                 list_dir=False,
    #                 suffix=self.img_suffix,
    #                 recursive=True,
    #                 backend_args=self.backend_args):
    #             data_info = dict(img_path=osp.join(img_dir, img))
    #             if ann_dir is not None:
    #                 seg_map = img.replace(self.img_suffix, "_pixels0" + self.seg_map_suffix)
    #                 data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
    #             data_info['label_map'] = self.label_map
    #             data_info['reduce_zero_label'] = self.reduce_zero_label
    #             data_info['seg_fields'] = []
    #             data_list.append(data_info)
    #         data_list = sorted(data_list, key=lambda x: x['img_path'])
    #     return data_list

