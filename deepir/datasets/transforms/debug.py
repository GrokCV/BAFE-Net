from mmcv.transforms import BaseTransform

from deepir.registry import TRANSFORMS

@TRANSFORMS.register_module()
class PrintPipeline(BaseTransform):
    """Resize images & bbox & seg.

    This transform resizes the input image according to ``scale`` or
    ``scale_factor``. Bboxes, masks, and seg map are then resized
    with the same scale factor.
    if ``scale`` and ``scale_factor`` are both set, it will use ``scale`` to
    resize.

    Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_masks (BitmapMasks | PolygonMasks) (optional)
    - gt_seg_map (np.uint8) (optional)

    Modified Keys:

    - img
    - img_shape
    - gt_bboxes
    - gt_masks
    - gt_seg_map


    Added Keys:

    - scale
    - scale_factor
    - keep_ratio
    - homography_matrix

    Args:
        scale (int or tuple): Images scales for resizing. Defaults to None
        scale_factor (float or tuple[float]): Scale factors for resizing.
            Defaults to None.
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image. Defaults to False.
        clip_object_border (bool): Whether to clip the objects
            outside the border of the image. In some dataset like MOT17, the gt
            bboxes are allowed to cross the border of images. Therefore, we
            don't need to clip the gt bboxes in these cases. Defaults to True.
        backend (str): Image resize backend, choices are 'cv2' and 'pillow'.
            These two backends generates slightly different results. Defaults
            to 'cv2'.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend. Defaults
            to 'bilinear'.
    """


    def transform(self, results: dict) -> dict:

        print(results)
        # if 'gt_seg_map' in results:
        #     for _ in range(5):
        #         print("'gt_seg_map' in results")
        # else:
        #     for _ in range(5):
        #         print("'gt_seg_map' not in results")
        # print("results['gt_bboxes']", results['gt_bboxes'])
        # print("type(results['gt_bboxes']):", type(results['gt_bboxes']))

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        return repr_str
