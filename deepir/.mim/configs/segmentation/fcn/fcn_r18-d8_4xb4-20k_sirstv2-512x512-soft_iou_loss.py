"""
Config file corresponding to the FCN model trained on SIRStv2 dataset.

Train Script:

CUDA_VISIBLE_DEVICES=0 python tools/train_seg.py configs/segmentation/fcn/fcn_r18-d8_4xb4-20k_sirstv2-512x512-soft_iou_loss.py

Test Script:

"""

_base_ = [
    '../_base_/models/fcn_r50-d8.py',
    '../_base_/datasets/sirstv2_seg.py',
    'mmseg::_base_/default_runtime.py',
    'mmseg::_base_/schedules/schedule_20k.py'
]

## data
# test_evaluator = dict(
#     type='IoUMetric',
#     iou_metrics=['mIoU'],
#     output_dir='work_dirs/format_results')

## model
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    data_preprocessor=data_preprocessor,
    # pretrained='https://download.openmmlab.com/pretrain/third_party/resnet18_v1c-b5776b93.pth',
    backbone=dict(
        type='ResNetV1c',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        in_channels=512,
        num_classes=2,
        out_channels=1,
        loss_decode=dict(
            type='deepir.SoftIoULoss',
            loss_name='loss_soft_iou',
            use_sigmoid=True,
            loss_weight=1.0)),
    auxiliary_head=dict(
        in_channels=256,
        num_classes=2,
        out_channels=1,
        loss_decode=dict(
            type='deepir.SoftIoULoss',
            loss_name='loss_aux_soft_iou',
            use_sigmoid=True,
            loss_weight=0.4)))

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

# training schedule
train_cfg = dict(type='IterBasedTrainLoop', max_iters=20000, val_interval=1000)
default_hooks = dict(visualization=dict(
    type='SegVisualizationHook', interval=1))