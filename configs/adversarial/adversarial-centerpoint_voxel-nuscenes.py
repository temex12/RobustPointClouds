"""
CORRECTED Adversarial configuration for CenterPoint on NuScenes.
This properly extends the voxel-based CenterPoint model with adversarial components.

Key fixes:
1. Uses AdversarialCenterPoint instead of AdversarialVoxelNet
2. Matches the exact voxel-based CenterPoint architecture 
3. Handles 5D NuScenes point cloud data correctly
4. Uses conservative parameters for stable training
"""
_base_ = [
    '../../mmdetection3d/configs/centerpoint/centerpoint_voxel01_second_secfpn_head-dcn-circlenms_8xb4-cyclic-20e_nus-3d.py'
]

custom_imports = dict(
    imports=[
        'models.detectors.adversarial_centerpoint', 
        'models.adversarial.voxel_perturber'
    ], 
    allow_failed_imports=False)

# Override model to use adversarial version
model = dict(
    type='AdversarialCenterPoint',  # Use the new CenterPoint-specific detector
    
    # Add adversarial components
    adversary_cfg=dict(
        type='VoxelPerturber',
        sensor_error_bound=0.2,  # Velodyne HDL-64E sensor error bound
        voxel_size=[0.1, 0.1, 0.2],  # Match baseline CenterPoint voxel size exactly
        use_spatial_attention=True,
        hidden_channels=[16, 32, 64]
    ),
    
    # Conservative adversarial training parameters
    adversarial_loss_weight=0.05,  # Start very small
    regularization_weight=0.005,   # Start very small
)

# Custom hooks for adversarial training
custom_hooks = [
    dict(
        type='L2NormRegularizationHook',
        regularization_strength=0.005,  # Reduced
        priority='NORMAL'),
    dict(
        type='EpochTrackerHook',
        priority='VERY_HIGH')
]

# Training settings - conservative approach
train_cfg = dict(
    by_epoch=True,
    max_epochs=20,
    val_interval=5)  # Validate less frequently

# Very conservative optimizer for adversarial training
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', 
        lr=0.00005,  # Much smaller learning rate
        weight_decay=0.001,  # Reduced weight decay
        betas=(0.9, 0.999),
        eps=1e-8
    ),
    clip_grad=dict(max_norm=1.0, norm_type=2)  # Strong gradient clipping
)

# Conservative learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=False,
        begin=0,
        end=2000),  # Longer warmup
    dict(
        type='CosineAnnealingLR',
        T_max=20,
        eta_min=0.000001,  # Very small minimum LR
        by_epoch=True,
        begin=0,
        end=20)
]

# Load from the correct pretrained checkpoint (same as baseline)
load_from = 'checkpoints/centerpoint_01voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus_20220810_052355-a6928835.pth'

# Logging
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', 
        interval=1,
        save_best='NuScenes metric/pred_instances_3d_NuScenes/mAP',
        rule='greater'
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook'))

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

# Environment settings
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

# Dataset settings
train_dataloader = dict(
    batch_size=1,  # Conservative batch size for single GPU training
    num_workers=4,  # Increased workers for better data loading
    persistent_workers=True)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True)

test_dataloader = val_dataloader