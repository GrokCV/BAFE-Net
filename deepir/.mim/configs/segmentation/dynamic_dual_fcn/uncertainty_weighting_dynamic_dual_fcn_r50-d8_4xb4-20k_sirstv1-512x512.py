"""
Config file corresponding to the FCN model trained on SIRStv1 dataset.

Train Script:

CUDA_VISIBLE_DEVICES=3 python tools/train_seg.py configs/dynamic_dual_fcn/uncertainty_weighting_dynamic_dual_fcn_r50-d8_4xb4-20k_sirstv1-512x512.py

Test Script:
"""

_base_ = [
    '../_base_/models/uncertainty_weighting_dynamic_dual_fcn_r50-d8.py',
    '../_base_/datasets/sirstv1_dual_seg.py',
    'mmseg::_base_/default_runtime.py',
    'mmseg::_base_/schedules/schedule_20k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=2, out_channels=1,
                     bg_num_classes=2, bg_out_channels=1),
    auxiliary_head=dict(num_classes=2, out_channels=1,
                        bg_num_classes=2, bg_out_channels=1))
# training schedule
# train_cfg = dict(val_interval=100)

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
    type='deepir.DualSegVisualizationHook', interval=1))

# runtime settings
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='deepir.DualSegLocalVisualizer', vis_backends=vis_backends, name='visualizer')