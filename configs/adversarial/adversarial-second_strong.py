
# Use our working SimpleAdversarialVoxelNet instead of complex AdversarialVoxelNet
_base_ = './adversarial-second_fixed.py'

# STABILIZED optimizer settings to prevent gradient explosion
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0009, betas=(0.9, 0.999), weight_decay=0.01),
    clip_grad=dict(max_norm=5, norm_type=2),  # Stronger gradient clipping
    paramwise_cfg=dict(
        custom_keys={
            'adversary': dict(lr_mult=1.0, decay_mult=1.0),  # Use base LR for adversary (paper's approach)
            'perturber': dict(lr_mult=1.0, decay_mult=1.0),
            'VoxelPerturber': dict(lr_mult=1.0, decay_mult=1.0),
        },
        bias_lr_mult=1.0,
        bias_decay_mult=0.0,
        norm_decay_mult=0.0,
        dwconv_decay_mult=0.0,
        bypass_duplicate=True
    ))

# BALANCED attack parameters - prevent collapse while being effective
model = dict(
    adversarial_loss_weight=0.9,    # Moderate increase from 0.7 to avoid collapse
    regularization_weight=0.01,     # Increase regularization to stabilize training
    adversary_cfg=dict(
        type='VoxelPerturber',
        sensor_error_bound=0.16,     # Reduce bounds to prevent explosion
        voxel_size=[0.05, 0.05, 0.1],
        use_spatial_attention=True,
        hidden_channels=[32, 64, 128]  # Keep moderate network size
    )
)

# CRITICAL: Override base config's 20-epoch setting
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=3,   # SHORT training to prevent adversarial adaptation
    val_interval=1  # Evaluate every epoch to catch peak effect
)

# Override scheduler to match 3-epoch training
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=500),
    dict(type='CosineAnnealingLR', T_max=3, eta_min=0.000001, by_epoch=True, begin=0, end=3)
]

# Paper's batch size: 6 (adjust if computationally feasible)
# Note: Current configs might have different batch settings in dataloader

# Save checkpoints every epoch for short training
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,   # Save every epoch for 3-epoch training
        max_keep_ckpts=3,  # Keep all 3 checkpoints
        save_best='auto',
        rule='greater'
    )
)
