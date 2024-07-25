# Copyright GrokCV. All rights reserved.
import os.path as osp
import xml.etree.ElementTree as ET
from typing import List, Optional, Union

import numpy as np

import mmcv
from mmengine.fileio import get, get_local_path, list_from_file
from mmdet.datasets.xml_style import XMLDataset

from deepir.registry import DATASETS


@DATASETS.register_module()
class SIRSTVOCDetDataset(XMLDataset):
    """Dataset for PASCAL VOC."""

    METAINFO = {
        "classes": ("Target",),
        # palette is a list of color tuples, which is used for visualization.
        "palette": [
            (106, 0, 228),
        ],
    }

    def __init__(self, img_subdir="PNGImages", ann_subdir="SIRST/BBox", **kwargs):
        super().__init__(img_subdir=img_subdir, ann_subdir=ann_subdir, **kwargs)
        self._metainfo["dataset_type"] = None

    def load_data_list(self) -> List[dict]:
        """Load annotation from XML style ann_file.

        Returns:
            list[dict]: Annotation info from XML file.
        """
        assert (
            self._metainfo.get("classes", None) is not None
        ), "`classes` in `XMLDataset` can not be None."
        self.cat2label = {cat: i for i, cat in enumerate(self._metainfo["classes"])}

        data_list = []
        img_ids = list_from_file(self.ann_file, backend_args=self.backend_args)
        for img_id in img_ids:
            file_name = osp.join(self.img_subdir, f"{img_id}.png")
            xml_path = osp.join(self.sub_data_root, self.ann_subdir, f"{img_id}.xml")

            raw_img_info = {}
            raw_img_info["img_id"] = img_id
            raw_img_info["file_name"] = file_name
            raw_img_info["xml_path"] = xml_path

            parsed_data_info = self.parse_data_info(raw_img_info)
            data_list.append(parsed_data_info)
        return data_list

    def _parse_instance_info(
        self, raw_ann_info: ET, minus_one: bool = True
    ) -> List[dict]:
        """parse instance information.

        Args:
            raw_ann_info (ElementTree): ElementTree object.
            minus_one (bool): Whether to subtract 1 from the coordinates.
                Defaults to True.

        Returns:
            List[dict]: List of instances.
        """
        instances = []
        for obj in raw_ann_info.findall("object"):
            instance = {}
            name = obj.find("name").text
            if name not in self._metainfo["classes"]:
                continue
            difficult = obj.find("difficult")
            difficult = 0 if difficult is None else int(difficult.text)
            bnd_box = obj.find("bndbox")
            if bnd_box is not None:
                bbox = [
                    int(float(bnd_box.find("xmin").text)),
                    int(float(bnd_box.find("ymin").text)),
                    int(float(bnd_box.find("xmax").text)),
                    int(float(bnd_box.find("ymax").text)),
                ]

                # VOC needs to subtract 1 from the coordinates
                if minus_one:
                    bbox = [x - 1 for x in bbox]

                ignore = False
                if self.bbox_min_size is not None:
                    assert not self.test_mode
                    w = bbox[2] - bbox[0]
                    h = bbox[3] - bbox[1]
                    if w < self.bbox_min_size or h < self.bbox_min_size:
                        ignore = True
                if difficult or ignore:
                    instance["ignore_flag"] = 1
                else:
                    instance["ignore_flag"] = 0
                instance["bbox"] = bbox
                instance["bbox_label"] = self.cat2label[name]
                instances.append(instance)
            else:
                instance["bbox"] = np.zeros((0, 4))
                instance["bbox_label"] = np.zeros((0,))
                instance["ignore_flag"] = 1
                instances.append(instance)
        return instances
