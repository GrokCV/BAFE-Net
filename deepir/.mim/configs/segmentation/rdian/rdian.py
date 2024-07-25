"""
Train Script:

CUDA_VISIBLE_DEVICES=3 python tools/train_seg.py configs/segmentation/rdian/rdian.py

Test Script:

"""

_base_ = [
    '../_base_/datasets/densesirst_seg.py',
    "../_base_/schedules/schedule_20k.py",
    "../_base_/default_runtime.py",
]

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
crop_size = (512, 512)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    size=crop_size,
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='deepir.RDIAN'),
    decode_head=dict(
        type='deepir._FCNHead',
        in_channels=32, #[64, 32, 16, 8]
        # in_index=4,
        in_index=1,
        channels=1,
        bias=False,
        drop=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
            # type='DiceLoss', loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

