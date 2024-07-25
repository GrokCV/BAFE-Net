"""
Train Script:

CUDA_VISIBLE_DEVICES=3 python tools/train_det.py configs/detection/fcos_changer_seg/fcos_changer_seg_r18-caffe_fpn_gn-head_1x_sirstv2.py --work-dir work_dirs/fcos_changer_seg_r18-caffe_fpn_gn-head_1x_sirstv2/test2

Test Script:

CUDA_VISIBLE_DEVICES=0 python tools/test_det.py configs/detection/fcos_changer_seg/fcos_changer_seg_r18-caffe_fpn_gn-head_1x_sirstv2.py work_dirs/fcos_changer_seg_r18-caffe_fpn_gn-head_1x_sirstv2/mAP07/mAP07-BAFE-Net+mlp_64-0284-0343/best_pascal_voc_mAP_epoch_5.pth
"""

_base_ = [
    # '../_base_/datasets/sirst_det_seg_voc.py',
    # '../_base_/datasets/sirst_det_seg_voc_cp.py',
    '../_base_/datasets/sirst_det_seg_voc_skycp.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

INF = 1e8
# model settings
model = dict(
    type='FCOS',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[102.9801, 115.9465, 122.7717],
        std=[1.0, 1.0, 1.0],
        bgr_to_rgb=False,
        pad_size_divisor=32,
        pad_seg=True),
    backbone=dict(
        type='ResNet',
        # depth=18,
        # depth=34,
        depth=50,
        # depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            # checkpoint='torchvision://resnet18')),
            # checkpoint='torchvision://resnet34')),
            checkpoint='torchvision://resnet50')),
            # checkpoint='torchvision://resnet101')),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 256, 512],
        # in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',  # use P5
        num_outs=3,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='deepir.FCOSChangerSegHead',
        num_classes=1,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        # strides=[8, 16, 32, 64, 128],
        strides=[8, 16, 32],
        regress_ranges=((-1, 64), (64, 128), (128, 256)),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='IoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        # loss_seg = dict(
        #     type='FocalLoss',
        #     use_sigmoid=True,
        #     gamma=2.0,
        #     alpha=0.25,
        #     loss_weight=1.0)
        loss_seg=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True, loss_weight=1.0, ignore_index=255),
        ),
    # testing settings
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))

# learning rate
param_scheduler = [
    dict(type='ConstantLR', factor=1.0 / 3, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=20,
        by_epoch=True,
        milestones=[8, 16],
        gamma=0.1)
]

# optimizer
# optim_wrapper = dict(
#     optimizer=dict(lr=0.01),
#     paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.),
#     clip_grad=dict(max_norm=35, norm_type=2))
# optimizer
base_lr = 1.0
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='DAdaptAdam', lr=base_lr, weight_decay=0.05,
        decouple=True),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))
