_base_ = [
    '../../mmdetection3d/configs/_base_/models/second_hv_secfpn_kitti.py',
    '../../mmdetection3d/configs/_base_/datasets/kitti-3d-3class.py',
    '../../mmdetection3d/configs/_base_/schedules/cyclic-40e.py',
    '../../mmdetection3d/configs/_base_/default_runtime.py'
]

# Import necessary modules
custom_imports = dict(
    imports=['models', 'models.detectors.adversarial_voxelnet', 'custom_hook', 'mmdet.models.losses'],
    allow_failed_imports=False)

# Model configuration with adversarial component
model = dict(
    type='AdversarialVoxelNet',
    # Copy voxel_encoder settings from base - EXACT MATCH
    voxel_encoder=dict(type='HardSimpleVFE'),
    # Copy middle_encoder settings from base - EXACT MATCH  
    middle_encoder=dict(
        type='SparseEncoder',
        in_channels=4,
        sparse_shape=[41, 1600, 1408],
        order=('conv', 'norm', 'act')),
    # Copy backbone settings from base - EXACT MATCH (FIXED: was [3,5] and [2,2])
    backbone=dict(
        type='SECOND',
        in_channels=256,
        layer_nums=[5, 5],  # FIXED: Must match baseline [5, 5] not [3, 5]
        layer_strides=[1, 2],  # FIXED: Must match baseline [1, 2] not [2, 2] 
        out_channels=[128, 256]),
    # Copy neck settings - EXACT MATCH to baseline
    neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        upsample_strides=[1, 2],
        out_channels=[256, 256]),
    # Update bbox_head for 3 classes
    bbox_head=dict(
        type='Anchor3DHead',
        num_classes=3,  # Car, Pedestrian, Cyclist
        in_channels=512,
        feat_channels=512,
        use_direction_classifier=True,
        anchor_generator=dict(
            type='Anchor3DRangeGenerator',
            ranges=[
                [0, -40.0, -0.6, 70.4, 40.0, -0.6],
                [0, -40.0, -0.6, 70.4, 40.0, -0.6],
                [0, -40.0, -0.6, 70.4, 40.0, -0.6],
            ],
            sizes=[[3.9, 1.6, 1.56], [0.8, 0.6, 1.73], [1.76, 0.6, 1.73]],
            rotations=[0, 1.57],
            reshape_out=False),
        diff_rad_by_sin=True,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(
            type='mmdet.SmoothL1Loss',
            beta=1.0 / 9.0,
            loss_weight=2.0),
        loss_dir=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=0.2)),
    # Adversarial configuration - BALANCED for 15-25% performance drop
    # Based on successful NuScenes configuration that achieved good results
    # Key changes from previous attempts:
    # - Much higher adversarial_loss_weight (0.01) similar to NuScenes (0.05)
    # - Minimal regularization to allow effective perturbations
    # - Larger network capacity for more expressive perturbations
    # - Sensor bound near maximum but still physically plausible
    adversary_cfg=dict(
        type='VoxelPerturber',
        sensor_error_bound=0.2,  # Maximum 20cm - Velodyne ±20cm limit
        voxel_size=[0.05, 0.05, 0.1],  # KITTI voxel size
        use_spatial_attention=True,  # Enable targeted perturbations
        hidden_channels=[64, 128, 64]),  # Even larger network for KITTI's harder task
    adversarial_loss_weight=0.1,  # Conservative base weight - will be dynamically adjusted
    regularization_weight=0.02,  # Moderate regularization for stability
    # model training and testing settings
    train_cfg=dict(
        assigner=[
            dict(  # for Car
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.6,
                neg_iou_thr=0.45,
                min_pos_iou=0.45,
                ignore_iof_thr=-1),
            dict(  # for Pedestrian
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.35,
                neg_iou_thr=0.2,
                min_pos_iou=0.2,
                ignore_iof_thr=-1),
            dict(  # for Cyclist
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.35,
                neg_iou_thr=0.2,
                min_pos_iou=0.2,
                ignore_iof_thr=-1),
        ],
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_thr=0.01,
        score_thr=0.1,
        min_bbox_size=0,
        nms_pre=100,
        max_num=50))

# Custom hooks for adversarial training
custom_hooks = [
    dict(type='EpochTrackerHook'),
    dict(type='NaNDetectionHook', max_nan_count=10)
]

# Modify training schedule for better convergence - BALANCED approach
# Separate optimizers for detector and adversary
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.001, eps=1e-8),
    clip_grad=dict(max_norm=0.5, norm_type=2),  # Stronger gradient clipping to prevent explosion
    # Custom parameter groups to handle adversary separately
    paramwise_cfg=dict(
        custom_keys={
            'adversary': dict(lr_mult=2.0)  # Adversary uses HIGHER LR to maintain attack strength
        }
    ))

# Use learning rate schedule with warmup for stability - ADJUSTED for 40 epochs
param_scheduler = [
    # Linear warmup for first 2000 iterations (doubled for longer training)
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=False,
        begin=0,
        end=2000),
    # Then cosine annealing over 30 epochs
    dict(
        type='CosineAnnealingLR',
        T_max=30,  # Match max_epochs
        eta_min=0.0001 * 0.01,  # 1% of base LR
        begin=0,
        end=30,  # Match max_epochs
        by_epoch=True,
        convert_to_iter_based=True)
]

# Training epochs - Adjusted for balanced adversarial training
# With reduced attack strength, we can train longer for better convergence
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=30, val_interval=3)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Load from pretrained baseline checkpoint (fixed tensor format)
load_from = 'checkpoints/second_hv_secfpn_8xb6-80e_kitti-3d-3class-fixed.pth'

# The base config already has proper data pipeline settings
# We only need to ensure the batch sizes are set correctly in the training script

# Visualization
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')