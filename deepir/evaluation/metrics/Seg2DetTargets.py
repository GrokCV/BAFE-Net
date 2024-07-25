import warnings
import numpy as np
from skimage import measure as skm
import torch
import torch.nn.functional as nnf
import mmcv

class NoCoTargets(object):
    """Generate the ground truths for NoCo TODO: paper title not decided yet

    Args:
    """
    def __init__(self, mu=0, sigma=1):
        assert isinstance(mu, float) or isinstance(mu, int)
        assert isinstance(sigma, float) or isinstance(sigma, int)
        assert sigma > 0
        self.mu = mu
        self.sigma = sigma

    def get_bboxes(self, gt_semantic_seg):
        bboxes = []
        gt_labels = skm.label(gt_semantic_seg, background=0)
        gt_regions = skm.regionprops(gt_labels)
        for props in gt_regions:
            ymin, xmin, ymax, xmax = props.bbox
            bboxes.append([xmin, ymin, xmax, ymax])
        return bboxes

    def generate_gt_noco_map(self, img, gt_bboxes):
        """Generate the ground truths for NoCo targets

        Args:
            img: np.ndarray with shape (H, W)
            gt_semantic_seg: np.ndarray with shape (H, W)
            gt_bboxes: list[list]
        """
        if len(img.shape) == 3:
            img = img.mean(-1)
        gt_noco_map = np.zeros_like(img).astype(float)
        if len(gt_bboxes) == 0:
            return gt_noco_map
        img_h, img_w = img.shape
        for _, gt_bbox in enumerate(gt_bboxes):
            # target cell coordinates
            tgt_cmin, tgt_rmin, tgt_cmax, tgt_rmax = gt_bbox
            tgt_cmin = int(tgt_cmin)
            tgt_rmin = int(tgt_rmin)
            tgt_cmax = int(tgt_cmax)
            tgt_rmax = int(tgt_rmax)
            # target cell size
            tgt_h = tgt_rmax - tgt_rmin
            tgt_w = tgt_cmax - tgt_cmin
            # Gaussian Matrix Size
            max_bg_hei = int(tgt_h * 3)
            max_bg_wid = int(tgt_w * 3)
            mesh_x, mesh_y = np.meshgrid(np.linspace(-1.5, 1.5, max_bg_wid),
                                         np.linspace(-1.5, 1.5, max_bg_hei))
            dist = np.sqrt(mesh_x*mesh_x + mesh_y*mesh_y)
            gaussian_like = np.exp(-((dist-self.mu)**2 / (2.0*self.sigma**2)))
            # unbounded background patch coordinates
            unb_bg_rmin = tgt_rmin - tgt_h
            unb_bg_rmax = tgt_rmax + tgt_h
            unb_bg_cmin = tgt_cmin - tgt_w
            unb_bg_cmax = tgt_cmax + tgt_w
            # bounded background patch coordinates
            bnd_bg_rmin = max(0, tgt_rmin - tgt_h)
            bnd_bg_rmax = min(tgt_rmax + tgt_h, img_h)
            bnd_bg_cmin = max(0, tgt_cmin - tgt_w)
            bnd_bg_cmax = min(tgt_cmax + tgt_w, img_w)
            # inds in Gaussian Matrix
            rmin_ind = bnd_bg_rmin - unb_bg_rmin
            cmin_ind = bnd_bg_cmin - unb_bg_cmin
            rmax_ind = max_bg_hei - (unb_bg_rmax - bnd_bg_rmax)
            cmax_ind = max_bg_wid - (unb_bg_cmax - bnd_bg_cmax)
            # distance weights
            bnd_gaussian = gaussian_like[rmin_ind:rmax_ind, cmin_ind:cmax_ind]

            # generate contrast weights
            tgt_cell = img[tgt_rmin:tgt_rmax, tgt_cmin:tgt_cmax]
            # max_tgt = tgt_cell.max().astype(float)
            # Check if tgt_cell is empty
            if tgt_cell.size == 0:
                # Handle the case where tgt_cell is empty
                # print(1)
                return gt_noco_map
            else:
                max_tgt = tgt_cell.max().astype(float)
            contrast_rgn = img[bnd_bg_rmin:bnd_bg_rmax, bnd_bg_cmin:bnd_bg_cmax]
            min_bg = contrast_rgn.min().astype(float)
            contrast_rgn = (contrast_rgn - min_bg) / (max_tgt - min_bg + 0.01)

            # fuse distance weights and contrast weights
            noco_rgn = bnd_gaussian * contrast_rgn
            max_noco = noco_rgn.max()
            min_noco = noco_rgn.min()
            noco_rgn = (noco_rgn - min_noco) / (max_noco - min_noco + 0.001)

            gt_noco_map[
                bnd_bg_rmin:bnd_bg_rmax, bnd_bg_cmin:bnd_bg_cmax] = noco_rgn

        return gt_noco_map

    def get_gt_noco_map(self, img, gt_semantic_seg):

        gt_bboxes = self.get_bboxes(gt_semantic_seg) # 为什么是gtbbox
        gt_noco_map = self.generate_gt_noco_map(img, gt_bboxes)
        # results['gt_noco_map'] = gt_noco_map
        # if "mask_fields" in results:
        #     results["mask_fields"].append("gt_noco_map")

        return gt_noco_map
    
    def get_det_gt_noco_map(self, img, gt_bboxes):

        gt_noco_map = self.generate_gt_noco_map(img, gt_bboxes)
        # results['gt_noco_map'] = gt_noco_map
        # if "mask_fields" in results:
        #     results["mask_fields"].append("gt_noco_map")

        return gt_noco_map

