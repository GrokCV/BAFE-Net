# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='deepir.DualSegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type='deepir.EncoderDualDecoder',
    data_preprocessor=data_preprocessor,
    pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='deepir.DynamicDualFCNHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        num_convs=2,
        concat_input=True,
        norm_cfg=norm_cfg,
        align_corners=False,
        # target head
        dropout_ratio=0.1,
        num_classes=2,
        out_channels=1,
        loss_decode=dict(
            type='CrossEntropyLoss',  loss_name='loss_ce',
            use_sigmoid=True, loss_weight=1.0),
        # background head
        bg_dropout_ratio=0.1,
        bg_num_classes=2,
        bg_out_channels=1,
        bg_loss_decode=dict(
            type='CrossEntropyLoss', loss_name='loss_bg_ce',
            use_sigmoid=True, loss_weight=1.0)),
    auxiliary_head=dict(
        type='deepir.DynamicDualFCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        norm_cfg=norm_cfg,
        concat_input=False,
        align_corners=False,
        # target head
        dropout_ratio=0.1,
        num_classes=2,
        out_channels=1,
        loss_decode=dict(
            type='CrossEntropyLoss', loss_name='loss_aux_ce',
            use_sigmoid=True, loss_weight=0.4),
        # background head
        bg_dropout_ratio=0.1,
        bg_num_classes=2,
        bg_out_channels=1,
        bg_loss_decode=dict(
            type='CrossEntropyLoss', loss_name='loss_aux_bg_ce',
            use_sigmoid=True, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
