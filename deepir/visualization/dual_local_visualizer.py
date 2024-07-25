# Copyright (c) GrokCV. All rights reserved.
from typing import Dict, List, Optional

import mmcv
import numpy as np
from mmengine.dist import master_only

from mmseg.structures import SegDataSample
from mmseg.visualization import SegLocalVisualizer

from deepir.registry import VISUALIZERS

@VISUALIZERS.register_module()
class DualSegLocalVisualizer(SegLocalVisualizer):
    """Dual Local Visualizer.

    Args:
        name (str): Name of the instance. Defaults to 'visualizer'.
        image (np.ndarray, optional): the origin image to draw. The format
            should be RGB. Defaults to None.
        vis_backends (list, optional): Visual backend config list.
            Defaults to None.
        save_dir (str, optional): Save file dir for all storage backends.
            If it is None, the backend storage will not save any data.
        classes (list, optional): Input classes for result rendering, as the
            prediction of segmentation model is a segment map with label
            indices, `classes` is a list which includes items responding to the
            label indices. If classes is not defined, visualizer will take
            `cityscapes` classes by default. Defaults to None.
        palette (list, optional): Input palette for result rendering, which is
            a list of color palette responding to the classes. Defaults to None.
        dataset_name (str, optional): `Dataset name or alias <https://github.com/open-mmlab/mmsegmentation/blob/main/mmseg/utils/class_names.py#L302-L317>`_
            visulizer will use the meta information of the dataset i.e. classes
            and palette, but the `classes` and `palette` have higher priority.
            Defaults to None.
        alpha (int, float): The transparency of segmentation mask.
                Defaults to 0.8.

    Examples:
        >>> import numpy as np
        >>> import torch
        >>> from mmengine.structures import PixelData
        >>> from mmseg.data import SegDataSample
        >>> from mmseg.engine.visualization import SegLocalVisualizer

        >>> seg_local_visualizer = SegLocalVisualizer()
        >>> image = np.random.randint(0, 256,
        ...                     size=(10, 12, 3)).astype('uint8')
        >>> gt_sem_seg_data = dict(data=torch.randint(0, 2, (1, 10, 12)))
        >>> gt_sem_seg = PixelData(**gt_sem_seg_data)
        >>> gt_seg_data_sample = SegDataSample()
        >>> gt_seg_data_sample.gt_sem_seg = gt_sem_seg
        >>> seg_local_visualizer.dataset_meta = dict(
        >>>     classes=('background', 'foreground'),
        >>>     palette=[[120, 120, 120], [6, 230, 230]])
        >>> seg_local_visualizer.add_datasample('visualizer_example',
        ...                         image, gt_seg_data_sample)
        >>> seg_local_visualizer.add_datasample(
        ...                        'visualizer_example', image,
        ...                         gt_seg_data_sample, show=True)
    """ # noqa

    def __init__(self,
                 name: str = 'visualizer',
                 image: Optional[np.ndarray] = None,
                 vis_backends: Optional[Dict] = None,
                 save_dir: Optional[str] = None,
                 classes: Optional[List] = None,
                 palette: Optional[List] = None,
                 dataset_name: Optional[str] = None,
                 alpha: float = 0.8,
                 **kwargs):
        super().__init__(name, image, vis_backends, save_dir, classes,
                         palette, dataset_name, alpha, **kwargs)
        self.alpha: float = alpha
        self.set_dataset_meta(palette, classes, dataset_name)

    @master_only
    def add_datasample(
            self,
            name: str,
            image: np.ndarray,
            data_sample: Optional[SegDataSample] = None,
            draw_gt: bool = True,
            draw_pred: bool = True,
            show: bool = False,
            wait_time: float = 0,
            # TODO: Supported in mmengine's Viusalizer.
            out_file: Optional[str] = None,
            step: int = 0) -> None:
        """Draw datasample and save to all backends.

        - If GT and prediction are plotted at the same time, they are
        displayed in a stitched image where the left image is the
        ground truth and the right image is the prediction.
        - If ``show`` is True, all storage backends are ignored, and
        the images will be displayed in a local window.
        - If ``out_file`` is specified, the drawn image will be
        saved to ``out_file``. it is usually used when the display
        is not available.

        Args:
            name (str): The image identifier.
            image (np.ndarray): The image to draw.
            gt_sample (:obj:`SegDataSample`, optional): GT SegDataSample.
                Defaults to None.
            pred_sample (:obj:`SegDataSample`, optional): Prediction
                SegDataSample. Defaults to None.
            draw_gt (bool): Whether to draw GT SegDataSample. Default to True.
            draw_pred (bool): Whether to draw Prediction SegDataSample.
                Defaults to True.
            show (bool): Whether to display the drawn image. Default to False.
            wait_time (float): The interval of show (s). Defaults to 0.
            out_file (str): Path to output file. Defaults to None.
            step (int): Global step value to record. Defaults to 0.
        """
        classes = self.dataset_meta.get('classes', None)
        palette = self.dataset_meta.get('palette', None)

        gt_tgt_img_data = None
        pred_tgt_img_data = None
        gt_bg_img_data = None
        pred_bg_img_data = None

        # draw target head results
        if draw_gt and data_sample is not None and 'gt_sem_seg' in data_sample:
            gt_tgt_img_data = image
            assert classes is not None, 'class information is ' \
                                        'not provided when ' \
                                        'visualizing semantic ' \
                                        'segmentation results.'
            gt_tgt_img_data = self._draw_sem_seg(gt_tgt_img_data,
                                             data_sample.gt_sem_seg, classes,
                                             palette)

        if (draw_pred and data_sample is not None
                and 'pred_sem_seg' in data_sample):
            pred_tgt_img_data = image
            assert classes is not None, 'class information is ' \
                                        'not provided when ' \
                                        'visualizing semantic ' \
                                        'segmentation results.'
            pred_tgt_img_data = self._draw_sem_seg(pred_tgt_img_data,
                                               data_sample.pred_sem_seg,
                                               classes, palette)

        # draw background head results
        if draw_gt and data_sample is not None and 'gt_bg_map' in data_sample:
            gt_bg_img_data = image
            assert classes is not None, 'class information is ' \
                                        'not provided when ' \
                                        'visualizing semantic ' \
                                        'segmentation results.'
            gt_bg_img_data = self._draw_sem_seg(gt_bg_img_data,
                                             data_sample.gt_bg_map, classes,
                                             palette)

        if (draw_pred and data_sample is not None
                and 'pred_bg_map' in data_sample):
            pred_bg_img_data = image
            assert classes is not None, 'class information is ' \
                                        'not provided when ' \
                                        'visualizing semantic ' \
                                        'segmentation results.'
            pred_bg_img_data = self._draw_sem_seg(pred_bg_img_data,
                                               data_sample.pred_bg_map,
                                               classes, palette)

        if gt_tgt_img_data is not None and pred_tgt_img_data is not None:
            drawn_tgt_img = np.concatenate((gt_tgt_img_data, pred_tgt_img_data), axis=1)
        elif gt_tgt_img_data is not None:
            drawn_tgt_img = gt_tgt_img_data
        else:
            drawn_tgt_img = pred_tgt_img_data

        if gt_bg_img_data is not None and pred_bg_img_data is not None:
            drawn_bg_img = np.concatenate((gt_bg_img_data, pred_bg_img_data), axis=1)
        elif gt_bg_img_data is not None:
            drawn_bg_img = gt_bg_img_data
        else:
            drawn_bg_img = pred_bg_img_data

        drawn_img = np.concatenate((drawn_tgt_img, drawn_bg_img), axis=1)

        if show:
            self.show(drawn_img, win_name=name, wait_time=wait_time)

        if out_file is not None:
            mmcv.imwrite(mmcv.rgb2bgr(drawn_img), out_file)
        else:
            self.add_image(name, drawn_img, step)
