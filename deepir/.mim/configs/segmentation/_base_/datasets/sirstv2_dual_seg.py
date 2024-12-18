# dataset settings
dataset_type = 'deepir.SIRSTDualSegDataset'
data_root = 'data/SIRSTdevkit'
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='deepir.LoadDualSegAnnotations'),
    dict(
        type='RandomResize',
        scale=(2048, 512),
        ratio_range=(0.5, 2.0),
        resize_type='deepir.DualSegResize',
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='deepir.RandomDualSegFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='deepir.PackDualSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='deepir.DualSegResize', scale=(2048, 512), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='deepir.LoadDualSegAnnotations'),
    dict(type='deepir.PackDualSegInputs')
]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='deepir.DualSegResize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='deepir.RandomDualSegFlip', prob=0., direction='horizontal'),
                dict(type='deepir.RandomDualSegFlip', prob=1., direction='horizontal')
            ], [dict(type='deepir.LoadDualSegAnnotations')], [dict(type='deepir.PackDualSegInputs')]
        ])
]
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='PNGImages', seg_map_path='SIRST/PaletteMask',
            bg_map_path='SkySeg/PaletteMask'),
        ann_file='Splits/trainval_v2.txt',
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='PNGImages', seg_map_path='SIRST/PaletteMask',
            bg_map_path='SkySeg/PaletteMask'),
        ann_file='Splits/test_v2.txt',
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = [
    # 目标分割指标
    dict(type='IoUMetric', iou_metrics=['mIoU']),
    # 天空分割指标
    dict(type='deepir.BG_IoUMetric', iou_metrics=['bg_mIoU']),
]
test_evaluator = [
    # 目标分割指标
    dict(type='IoUMetric', iou_metrics=['mIoU']),
    # 天空分割指标
    dict(type='deepir.BG_IoUMetric', iou_metrics=['bg_mIoU'])
]
