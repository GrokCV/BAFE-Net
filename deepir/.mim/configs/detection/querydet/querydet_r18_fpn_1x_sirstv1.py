"""
Config file corresponding to the FCOS model trained on SIRStv1 dataset.

Train Script:

CUDA_VISIBLE_DEVICES=0 python tools/train_det.py configs/detection/querydet/querydet_r18_fpn_1x_sirstv1.py

Test Script:

"""

_base_ = [
    '../_base_/datasets/sirst_det_voc.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
    # './retinanet_tta.py'
]

train_dataloader = dict(
    dataset=dict(
        dataset=dict(ann_file='Splits/trainval_v1.txt')))

val_dataloader = dict(
    dataset=dict(
        dataset=dict(ann_file='Splits/test_v1.txt')))

# model settings
model = dict(
    type='QueryDet',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 256, 512],
        out_channels=64,
        start_level=0,
        add_extra_convs='on_input',
        num_outs=6),
    bbox_head=dict(
        type='QueryDetHead',
        num_classes=1,
        in_channels=64,
        stacked_convs=4,
        feat_channels=256,
        small_obj_scale = [[0, 32], [0, 64]],
        query_threshold = 0.15,
        query_context = 2,
        anchor_generator=dict(
            type='deepir.AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        query_loss=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        sampler=dict(
            type='PseudoSampler'),  # Focal loss should use PseudoSampler
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))
# optimizer
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))
