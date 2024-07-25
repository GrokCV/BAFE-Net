from typing import List, Sequence
from mmengine.evaluator import BaseMetric
import os.path as osp
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence
import torch
import numpy as np
import torchvision.transforms as transforms
from skimage import measure as skm
from mmengine.dist import is_main_process
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from mmengine.utils import mkdir_or_exist
from PIL import Image
from deepir.evaluation.metrics.mean_nocoap import eval_mnocoap
from deepir.evaluation.metrics.Seg2DetTargets import NoCoTargets
from prettytable import PrettyTable
from deepir.registry import METRICS


@METRICS.register_module()
class mNoCoAP_det_Metric(BaseMetric):
    """
    The metric first processes each batch of data_samples and predictions,
    and appends the processed results to the results list. Then it
    collects all results together from all ranks if distributed training
    is used. Finally, it computes the metrics of the entire dataset.
    """

    def __init__(
        self,
        ignore_index: int = 255,
        metrics: List[str] = ["mNoCoAP"],
        nan_to_num: Optional[int] = None,
        beta: int = 1,
        collect_device: str = "cpu",
        output_dir: Optional[str] = None,
        format_only: bool = False,
        prefix: Optional[str] = None,
        gt_noco_map_loader_cfg=None,
        **kwargs,
    ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

        self.ignore_index = ignore_index
        self.metrics = metrics
        self.nan_to_num = nan_to_num
        self.beta = beta
        self.output_dir = output_dir
        if self.output_dir and is_main_process():
            mkdir_or_exist(self.output_dir)
        self.format_only = format_only
        self.best_mnocoap = -np.inf

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data and data_samples.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        num_classes = len(self.dataset_meta["classes"])
        # 定义一个转换对象，用于将图像转换为单通道灰度图
        gray_transform = transforms.Grayscale(num_output_channels=1)
        inputs = data_batch["inputs"]
        # for data_img, data_sample in zip(data_batch, data_samples):
        noco_targets_instance = NoCoTargets()
        for input, data_sample in zip(inputs, data_samples):
            input = gray_transform(input).squeeze().cpu().numpy()
            gt_bboxes = data_sample["gt_instances"]["bboxes"].squeeze().cpu().numpy()
            pred_scores = (
                data_sample["pred_instances"]["scores"].squeeze().cpu().numpy()
            )
            pred_bboxes = (
                data_sample["pred_instances"]["bboxes"].squeeze().cpu().numpy()
            )
            # pred_labels = data_sample['pred_instances']['labels'].squeeze().cpu().numpy()
            # format_only always for test dataset without ground truth
            if not self.format_only:
                # label = data_sample['gt_sem_seg']['data'].squeeze().cpu().numpy() #.to(pred_label)
                det_centroids = det_results_to_noco_centroids(pred_bboxes, pred_scores)
                # 获取每个样本的非重叠区域地面真实分割图和边界框
                gt_bboxes = det_gt_bbox(gt_bboxes)
                gt_noco_map = noco_targets_instance.get_det_gt_noco_map(
                    input, gt_bboxes
                )

                self.results.append(
                    {
                        "det_centroids": det_centroids,
                        "gt_noco_map": gt_noco_map,
                        "gt_bbox": gt_bboxes,
                    }
                )
            # if self.output_dir is not None:
            #     basename = osp.splitext(osp.basename(
            #         data_sample['img_path']))[0]
            #     png_filename = osp.abspath(
            #         osp.join(self.output_dir, f'{basename}.png'))
            #     output_mask = pred_label.cpu().numpy()
            #     # The index range of official ADE20k dataset is from 0 to 150.
            #     # But the index range of output is from 0 to 149.
            #     # That is because we set reduce_zero_label=True.
            #     if data_sample.get('reduce_zero_label', False):
            #         output_mask = output_mask + 1
            #     output = Image.fromarray(output_mask.astype(np.uint8))
            #     output.save(png_filename)

    def compute_metrics(self, results: list) -> dict:
        """Evaluation on mNoCoAP.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            logger (logging.Logger | str | None): Logger used for printing
                    related information during evaluation. Default: None.
            noco_thrs (Sequence[float], optional): NoCo threshold used for
                evaluating recalls/mNoCoAPs. If set to a list, the average of
                all NoCos will also be computed. If not specified, [0.1, 0.2,
                0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] will be used.
                Default: None.

            Returns:
                dict
        """
        noco_thrs = np.linspace(
            0.1, 0.9, int(np.round((0.9 - 0.1) / 0.1)) + 1, endpoint=True
        )
        noco_thrs = [noco_thrs] if isinstance(noco_thrs, float) else noco_thrs

        # prepare inputs for eval_mnocoap
        # det_centroids = [result[0] for result in results]
        # gt_noco_maps = [self.get_gt_noco_map_by_idx(i)
        #                 for i in range(len(self))]
        # gt_bboxes = [self.get_gt_bbox_by_idx(i) for i in range(len(self))]
        det_centroids = []
        gt_noco_maps = []
        gt_bboxes = []

        # 迭代 self.results 并提取每个结果中的数据
        for result in self.results:
            det_centroids.append(result["det_centroids"])
            gt_noco_maps.append(result["gt_noco_map"])
            gt_bboxes.append(result["gt_bbox"])

        eval_results = OrderedDict()
        mean_nocoaps = []
        for noco_thr in noco_thrs:
            print_log(f'\n{"-" * 15}noco_thr: {noco_thr}{"-" * 15}')
            mean_nocoap, _ = eval_mnocoap(
                det_centroids, gt_noco_maps, gt_bboxes, noco_thr=noco_thr, logger="current"
            )
            mean_nocoaps.append(mean_nocoap)
            eval_results[f"NoCoAP{int(noco_thr * 100):02d}"] = round(mean_nocoap, 3)
        eval_results["mNoCoAP"] = sum(mean_nocoaps) / len(mean_nocoaps)
        print("eval_results['mNoCoAP']:", eval_results["mNoCoAP"])
        if self.best_mnocoap < eval_results["mNoCoAP"]:
            self.best_mnocoap = eval_results["mNoCoAP"]
        print("best eval_results['mNoCoAP']:", self.best_mnocoap)
        print_log(f"\n best eval_results['mNoCoAP']: {self.best_mnocoap}", logger="current")

        return eval_results

    def seg2centroid(self, pred, score_thr=0.5):
        """Convert pred to centroid detection results
        Args:
            pred (np.ndarray): shape (1, H, W)

        Returns:
            det_centroids (np.ndarray): shape (num_dets, 3)
        """

        if pred.ndim == 3:
            pred = pred.squeeze(0)
        # seg_mask = (pred > score_thr).astype(int)
        seg_mask = pred.copy()
        seg_mask[seg_mask > 0] = 1
        gt_labels = skm.label(seg_mask, background=0)
        gt_regions = skm.regionprops(gt_labels)
        centroids = []
        for props in gt_regions:
            ymin, xmin, ymax, xmax = props.bbox
            tgt_pred = pred[ymin:ymax, xmin:xmax]
            ridx, cind = np.unravel_index(
                np.argmax(tgt_pred, axis=None), tgt_pred.shape
            )
            tgt_score = tgt_pred[ridx, cind]
            centroids.append((xmin + cind, ymin + ridx, tgt_score))
            # centroids.append((xmin + cind, ymin + ridx))
        if len(centroids) == 0:
            return np.zeros((0, 3), dtype=np.float32)
        else:
            return np.array(centroids, dtype=np.float32)

    def det2centroid(self, pred, score_thr=0.5):
        """Convert pred to centroid detection results
        Args:
            pred (np.ndarray): shape (1, H, W)

        Returns:
            det_centroids (np.ndarray): shape (num_dets, 3)
        """

        if pred.ndim == 3:
            pred = pred.squeeze(0)
        # seg_mask = (pred > score_thr).astype(int)
        seg_mask = pred.copy()
        seg_mask[seg_mask > 0] = 1
        gt_labels = skm.label(seg_mask, background=0)
        gt_regions = skm.regionprops(gt_labels)
        centroids = []
        for props in gt_regions:
            ymin, xmin, ymax, xmax = props.bbox
            tgt_pred = pred[ymin:ymax, xmin:xmax]
            ridx, cind = np.unravel_index(
                np.argmax(tgt_pred, axis=None), tgt_pred.shape
            )
            tgt_score = tgt_pred[ridx, cind]
            centroids.append((xmin + cind, ymin + ridx, tgt_score))
            # centroids.append((xmin + cind, ymin + ridx))
        if len(centroids) == 0:
            return np.zeros((0, 3), dtype=np.float32)
        else:
            return np.array(centroids, dtype=np.float32)


def get_gt_bbox(gt_map):
    """Get gt bboxes for evaluation."""
    label_img = skm.label(gt_map, background=0)
    regions = skm.regionprops(label_img)
    bboxes = []
    for region in regions:
        ymin, xmin, ymax, xmax = region.bbox
        bboxes.append([xmin, ymin, xmax, ymax])
    if len(bboxes) == 0:
        return np.zeros((0, 4))
    else:
        return np.array(bboxes)


def det_gt_bbox(gt_bboxes):
    """Get gt bboxes for evaluation."""
    bboxes = []
    # 先检查列表是否为空
    if len(gt_bboxes) == 0:
        return np.zeros((0, 4))

    # 检查是单个还是多个边界框，并处理
    if isinstance(gt_bboxes[0], list) or isinstance(gt_bboxes[0], np.ndarray):
        # 处理多个边界框的情况
        for region in gt_bboxes:
            ymin, xmin, ymax, xmax = region
            bboxes.append([xmin, ymin, xmax, ymax])
    else:
        # 处理单个边界框的情况
        ymin, xmin, ymax, xmax = gt_bboxes
        bboxes.append([xmin, ymin, xmax, ymax])

    return np.array(bboxes)


def postprocess_result(seg_logits):
    # 第一步: 沿通道维度找出最大值的索引
    max_indices = seg_logits.argmax(dim=0)

    # 创建一个用于收集最大值的张量，初始化为0
    max_values = torch.zeros(
        (max_indices.shape[0], max_indices.shape[1]), dtype=torch.float32
    )

    # 遍历每个像素位置
    for i in range(max_indices.shape[0]):
        for j in range(max_indices.shape[1]):
            # 检查最大值索引是否为0
            if max_indices[i, j] != 0:
                # 提取最大值
                max_values[i, j] = seg_logits[max_indices[i, j], i, j]

    # 现在 max_values 包含了每个位置的最大值（索引非零的情况下），形状为 (H, W)
    # # 找出所有非零值
    # non_zero_values = max_values[max_values != 0]
    #
    # # 如果存在非零值，则进行归一化
    # if len(non_zero_values) > 0:
    #     # 找出非零值的最小值和最大值
    #     min_val = torch.min(non_zero_values)
    #     max_val = torch.max(non_zero_values)
    #
    #     # 计算非零值的范围
    #     range_val = max_val - min_val
    #
    #     # 只对非零值进行归一化
    #     # 这里使用一个掩码来选择非零元素，并原地更新这些元素的值
    #     max_values[max_values != 0] = (max_values[max_values != 0] - min_val) / range_val
    return max_values


# def det_results_to_noco_centroids1(bbox_results, pred_scores):
#     """Transform bbox_results to centroid_results for `evaluate` in
#     SIRSTDet2NoCoDataset.

#     Args:
#         bbox_results (list[list[np.ndarray]])

#     Returns:
#         centroid_results (list[np.ndarray]):
#     """

#     centroid_results = []
#     # for img_list in bbox_results:
#     for bboxes, pred_score in zip(bbox_results, pred_scores):
#         img_centroids = []
#         for bbox in bboxes:
#             if bbox.shape[0] == 0:
#                 # list[np.ndarray]
#                 img_centroids.append(np.zeros((0, 3), dtype=np.float32))
#             else:
#                 ymins = bbox[:, 0, None]
#                 xmins = bbox[:, 1, None]
#                 ymaxs = bbox[:, 2, None]
#                 xmaxs = bbox[:, 3, None]
#                 xcs = (xmins + xmaxs) / 2
#                 ycs = (ymins + ymaxs) / 2
#                 scores = pred_score
#                 # np.ndarray
#                 centroid = np.concatenate((xcs, ycs, scores), axis=1)
#                 # list[np.ndarray]
#                 img_centroids.append(centroid)
#         # np.ndarray
#         img_centroids = np.concatenate(img_centroids, axis=0)
#         # list[np.ndarray]
#         centroid_results.append(img_centroids)

#     return centroid_results

# def det_results_to_noco_centroids1(bbox_results, pred_scores):
#     """Transform bbox_results to centroid_results for `evaluate` in
#     SIRSTDet2NoCoDataset.

#     Args:
#         bbox_results (list[list[np.ndarray]]): Expected shape [n, 4] where each sublist contains [xmins, ymins, xmaxs, ymaxs]
#         pred_scores (list[np.ndarray]): List of prediction scores

#     Returns:
#         centroid_results (np.ndarray): Centroids and their scores
#     """
#     centroid_results = []
#     scores=[]
#     bboxes=[]
#     bboxes.append(bbox_results)
#     scores.append(pred_scores)
#     if bbox_results.shape[0] == 0:
#         return np.zeros((0, 3), dtype=np.float32)

#     # 处理边界框和分数
#     for bbox, pred_score in zip(bboxes, scores):
#         # if isinstance(bbox, np.ndarray) and bbox.ndim < 2:
#         #     continue  # 如果bbox格式不正确，则跳过此条目
#         ymins = bbox[0]
#         xmins = bbox[1]
#         ymaxs = bbox[2]
#         xmaxs = bbox[3]
#         xcs = (xmins + xmaxs) / 2
#         ycs = (ymins + ymaxs) / 2
#         scores = pred_score
#         centroid_results.append(np.column_stack((xcs, ycs, scores)))

#     if len(centroid_results) > 0:
#         return np.vstack(centroid_results)
#     else:
#         return np.zeros((0, 3), dtype=np.float32)


def det_results_to_noco_centroids(bbox_results, pred_scores):
    """Get gt bboxes for evaluation."""
    centroid_results = []

    # 先检查列表是否为空
    if len(bbox_results) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    # 检查是单个还是多个边界框，并处理
    if isinstance(bbox_results[0], list) or isinstance(bbox_results[0], np.ndarray):
        # 处理多个边界框的情况
        for bbox, score in zip(bbox_results, pred_scores):
            ymin, xmin, ymax, xmax = bbox
            xcs = (xmin + xmax) / 2
            ycs = (ymin + ymax) / 2
            centroid_results.append([xcs, ycs, score])
    else:
        # 处理单个边界框的情况
        ymin, xmin, ymax, xmax = bbox_results
        xcs = (xmin + xmax) / 2
        ycs = (ymin + ymax) / 2
        centroid_results.append([xcs, ycs, pred_scores])

    if len(centroid_results) > 0:
        return np.array(centroid_results, dtype=np.float32)
    else:
        return np.zeros((0, 3), dtype=np.float32)
