# Enhanced KITTI adversarial config with dynamic scaling and curriculum learning
# Designed to train STRONGER adversarial perturbations that don't weaken over epochs

# Use working KITTI SECOND config as base
_base_ = [
    '../../mmdetection3d/configs/_base_/models/second_hv_secfpn_kitti.py',
    '../../mmdetection3d/configs/_base_/datasets/kitti-3d-3class.py',
    '../../mmdetection3d/configs/_base_/schedules/cyclic-40e.py',
    '../../mmdetection3d/configs/_base_/default_runtime.py'
]

# Import custom modules
custom_imports = dict(
    imports=['models.detectors.enhanced_adversarial_voxelnet'], 
    allow_failed_imports=False
)

# Enhanced model with dynamic adversarial scaling
model = dict(
    type='StrongAdversarialVoxelNet',  # Use the enhanced detector
    
    # Enhanced adversarial training parameters
    adversarial_loss_weight=0.5,        # Start higher since we'll scale dynamically
    regularization_weight=0.005,        # Lower to allow stronger attacks
    
    # Dynamic scaling parameters - KEY INNOVATION
    dynamic_scaling=True,                # Enable dynamic strength scaling
    curriculum_learning=True,            # Enable curriculum learning
    scaling_factor=1.5,                  # Increase strength by 50% per epoch
    max_scaling=4.0,                     # Allow up to 4x strength increase
    momentum_alpha=0.9,                  # High momentum to prevent rapid weakening
    anti_adaptation_prob=0.15,           # 15% chance to skip detector updates
    
    # Advanced adversarial configuration (using regular VoxelPerturber for now)
    adversary_cfg=dict(
        type='VoxelPerturber',           # Use regular perturber for now
        sensor_error_bound=0.18,         # Slightly increased bounds
        voxel_size=[0.05, 0.05, 0.1],
        use_spatial_attention=True,
        hidden_channels=[64, 128, 256, 128],  # Deeper network for more complex perturbations
    ),
    
    # Class-specific attack parameters (enhanced)
    class_attack_weights=dict(
        Car=1.2,        # Increased from 1.0 - KITTI Cars are very robust
        Pedestrian=2.5, # Increased from 2.0 - Target pedestrian detection more aggressively
        Cyclist=1.8     # Increased from 1.5 - Boost cyclist attacks
    ),
    
    # Enhanced post-encoding noise scales
    post_encoding_noise_scales=dict(
        Car=0.3,        # Increased from 0.2
        Pedestrian=0.5, # Increased from 0.3
        Cyclist=0.4,    # Increased from 0.25
        default=0.3     # Increased default
    ),
    
    # KITTI 3-class bbox head configuration (override base config)
    bbox_head=dict(
        type='Anchor3DHead',
        num_classes=3,  # Car, Pedestrian, Cyclist
        anchor_generator=dict(
            _delete_=True,
            type='Anchor3DRangeGenerator',
            ranges=[
                [0, -40.0, -1.78, 70.4, 40.0, -1.78],  # Car
                [0, -40.0, -0.6, 70.4, 40.0, -0.6],    # Pedestrian
                [0, -40.0, -0.6, 70.4, 40.0, -0.6]     # Cyclist
            ],
            sizes=[
                [3.9, 1.6, 1.56],  # Car
                [0.8, 0.6, 1.73],  # Pedestrian
                [1.76, 0.6, 1.73]  # Cyclist
            ],
            rotations=[0, 1.57],
            reshape_out=True
        )
    ),
    
    # Training configuration with single assigner (not list for 3-class)
    train_cfg=dict(
        _delete_=True,
        assigner=dict(
            type='Max3DIoUAssigner',
            iou_calculator=dict(type='BboxOverlapsNearest3D'),
            pos_iou_thr=0.5,
            neg_iou_thr=0.35,
            min_pos_iou=0.35,
            ignore_iof_thr=-1
        ),
        allowed_border=0,
        pos_weight=-1,
        debug=False
    ),
    
    # Test configuration for validation/prediction
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_thr=0.01,
        score_thr=0.1,
        min_bbox_size=0,
        nms_pre=100,
        max_num=50
    )
)

# Enhanced optimizer with learning rate scheduling for adversarial components
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.001, betas=(0.9, 0.999), weight_decay=0.01),
    clip_grad=dict(max_norm=15, norm_type=2),  # Increased gradient clipping
    paramwise_cfg=dict(
        custom_keys={
            # Higher learning rates for adversarial components
            'adversary': dict(lr_mult=2.0, decay_mult=0.5),      # 2x LR, less decay
            'perturber': dict(lr_mult=2.0, decay_mult=0.5),      # Adversary learns faster
            'VoxelPerturber': dict(lr_mult=2.0, decay_mult=0.5),
            # Slower learning for detector to prevent rapid adaptation
            'backbone': dict(lr_mult=0.5, decay_mult=1.5),       # Detector learns slower
            'bbox_head': dict(lr_mult=0.5, decay_mult=1.5),      # Prevent quick adaptation
        },
        bias_lr_mult=1.0,
        bias_decay_mult=0.0,
        norm_decay_mult=0.0,
        dwconv_decay_mult=0.0,
        bypass_duplicate=True
    ))

# Longer training to allow adversarial strength to build up
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=10,   # Increased from 5 - allow more time for scaling
    val_interval=2   # Validate every 2 epochs
)

# Enhanced parameter scheduler with adversarial curriculum
param_scheduler = [
    # Warm-up phase
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=500),
    
    # Main training with cosine annealing
    dict(type='CosineAnnealingLR', T_max=10, eta_min=0.00001, by_epoch=True, begin=0, end=10),
    
    # Adversarial-specific scheduler (custom implementation would go here)
    # For now, the dynamic scaling in the model handles this
]

# Enhanced logging to track adversarial progression
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=2,      # Save every 2 epochs
        max_keep_ckpts=8,
        save_best='auto',
        rule='greater'
    ),
    logger=dict(
        type='LoggerHook',
        interval=25      # More frequent logging to track adversarial dynamics
    )
)

# Enhanced custom hooks for adversarial training
custom_hooks = [
    dict(
        type='EpochTrackerHook',  # Essential for epoch-aware adversarial training
        priority=50
    ),
    # Future: Add AdversarialProgressHook to track and adjust scaling
]

# Training data configuration - use batch size 1 to avoid collation issues
train_dataloader = dict(
    batch_size=1,     # Use batch size 1 to avoid tensor size mismatch issues
    num_workers=0,    # Disable multiprocessing to avoid worker issues
    sampler=dict(type='DefaultSampler', shuffle=True),
    persistent_workers=False
)

# Validation configuration
val_dataloader = dict(
    batch_size=1,
    num_workers=0,    # Disable multiprocessing for validation too
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=False
)

# Additional training arguments
train_pipeline_cfg = dict(
    # Enable mixed precision for faster training
    fp16=dict(loss_scale=512.),
    
    # Enhanced data augmentation to improve robustness
    data_augmentation=dict(
        enable_mixup=True,
        mixup_alpha=0.2,
        enable_cutout=True,
        cutout_ratio=0.1
    )
)

# Work directory
work_dir = './work_dirs/kitti_enhanced_stronger_adversarial'

# Visualization config for tracking adversarial progression
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='TensorboardVisBackend',
        save_dir='work_dirs/kitti_enhanced_stronger_adversarial/tensorboard'
    )
]
visualizer = dict(
    type='Det3DLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)

# Enhanced evaluation metrics
val_evaluator = dict(
    type='KittiMetric',
    ann_file='data/kitti/kitti_infos_val.pkl',
    metric='bbox',
    backend_args=None
)