"""
Config file corresponding to the FCN model trained on SIRStv1 dataset.

Train Script:

CUDA_VISIBLE_DEVICES=0 python tools/train_seg.py configs/fcn/fcn_r50-d8_4xb4-20k_sirstv1-512x512.py

Test Script:

python tools/test_seg.py configs/fcn/fcn_r50-d8_4xb4-20k_sirstv1-512x512.py \
    work_dirs/fcn_r50-d8_4xb4-20k_sirstv1-512x512/iter_20000.pth \
    --show-dir work_dirs/fcn_r50-d8_4xb4-20k_sirstv1-512x512/show
"""

_base_ = [
    '../_base_/models/fcn_r50-d8.py',
    '../_base_/datasets/sirstv1_seg.py',
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
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=2, out_channels=1),
    auxiliary_head=dict(num_classes=2, out_channels=1))

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
default_hooks = dict(visualization=dict(
    type='SegVisualizationHook', interval=1))