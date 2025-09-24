"""
Training script for KITTI 3-class adversarial model.
"""
import os
import sys

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Optional: Set CUDA device (uncomment to use specific GPU)
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # Use GPU 2

# Fix numpy compatibility
try:
    import fix_numpy_compat
    print("Numpy compatibility fixes applied")
except Exception as e:
    print(f"Warning: Could not apply numpy compatibility fixes: {e}")

# Ensure CUDA is enabled for evaluation metrics
if 'NUMBA_DISABLE_CUDA' in os.environ:
    del os.environ['NUMBA_DISABLE_CUDA']
    print("Enabled CUDA for evaluation metrics")

# Fix yapf compatibility issue
try:
    from fix_yapf_issue import patch_mmengine_config
    patch_mmengine_config()
    print("Successfully patched mmengine Config.pretty_text")
except Exception as e:
    print(f"Warning: Could not apply yapf patch: {e}")

# More robust yapf fix
import mmengine.config

def _fake_pretty_text(self):
    """Return config as string without yapf formatting."""
    return str(self._cfg_dict)

mmengine.config.Config.pretty_text = property(_fake_pretty_text)
print("Successfully overridden Config.pretty_text to bypass yapf")

# Import custom hooks to register them
try:
    import custom_hook
    print("Custom hooks registered successfully")
except Exception as e:
    print(f"Warning: Could not import custom hooks: {e}")

# Fix adversary optimizer issue
try:
    import fix_adversary_optimizer
    print("Adversary optimizer patch applied successfully")
except Exception as e:
    print(f"Warning: Could not apply adversary optimizer patch: {e}")

def main():
    """Train the adversarial model on KITTI 3-class dataset."""
    import argparse
    parser = argparse.ArgumentParser(description='Train KITTI 3-class adversarial model')
    parser.add_argument('--validate', action='store_true', default=True,
                       help='Whether to evaluate during training (default: True)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from latest checkpoint')
    parser.add_argument('--auto-resume', action='store_true', default=False,
                       help='Automatically resume if checkpoint exists (default: False, disabled due to architecture changes)')
    parser.add_argument('--freeze-detector', action='store_true',
                       help='Freeze detector components and only train adversary')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("TRAINING ADVERSARIAL MODEL ON KITTI 3-CLASS DATASET")
    print("=" * 80)
    print("Classes: Car, Pedestrian, Cyclist")
    print("=" * 80)
    
    # Configuration - Check for stronger config first, then fall back to fixed
    stronger_config = 'configs/adversarial/adversarial-second_stronger.py'
    fixed_config = 'configs/adversarial/adversarial-second_fixed.py'
    
    if os.path.exists(stronger_config):
        config = stronger_config
        print("Using STRONGER adversarial config (8x bounds, direct loss)")
    elif os.path.exists(fixed_config):
        config = fixed_config
        print("Using fixed adversarial config (4x bounds)")
    else:
        print("ERROR: No adversarial config found")
        return False
    # Set work directory based on config used
    if 'stronger' in config:
        work_dir = 'work_dirs/kitti_3class_adversarial_stronger'
    else:
        work_dir = 'work_dirs/kitti_3class_adversarial_fixed'
    
    # Check if config exists
    if not os.path.exists(config):
        print(f"ERROR: Config file not found: {config}")
        return False
    
    # Training arguments - start fresh (no resume flags)
    train_args = [
        config,
        '--work-dir', work_dir
        # Note: Not adding any resume-related flags to ensure fresh start
    ]
    
    # Handle validation flag - mmdetection3d uses --no-validate to disable
    if not args.validate:
        train_args.append('--no-validate')
        print("Validation disabled during training")
    else:
        print("Validation enabled during training (every 5 epochs)")
    
    # Check for existing checkpoints and handle incompatible checkpoints
    import glob
    checkpoint_dir = os.path.join(work_dir, '*.pth')
    checkpoints = glob.glob(checkpoint_dir)
    
    # Also check for log files that might interfere
    log_files = glob.glob(os.path.join(work_dir, '*.log')) + glob.glob(os.path.join(work_dir, '*.log.json'))
    
    # IMPORTANT: Due to architecture and optimizer changes, old checkpoints are incompatible
    # We need to either start fresh or only load model weights (not optimizer state)
    if checkpoints or log_files:
        if checkpoints:
            print(f"Found existing checkpoints: {[os.path.basename(cp) for cp in checkpoints]}")
        if log_files:
            print(f"Found existing log files: {[os.path.basename(lf) for lf in log_files]}")
        print("WARNING: Configuration has been updated for balanced attack strength.")
        print("Starting fresh training with new adversarial parameters.")
        
        # Backup old checkpoints and logs to avoid conflicts
        backup_dir = os.path.join(work_dir, 'old_checkpoints')
        os.makedirs(backup_dir, exist_ok=True)
        
        # Move checkpoint files
        import shutil
        for cp in checkpoints:
            backup_path = os.path.join(backup_dir, os.path.basename(cp))
            if not os.path.exists(backup_path):
                shutil.move(cp, backup_path)
                print(f"Moved {os.path.basename(cp)} to old_checkpoints/")
        
        # Move log files  
        for lf in log_files:
            backup_path = os.path.join(backup_dir, os.path.basename(lf))
            if not os.path.exists(backup_path):
                shutil.move(lf, backup_path)
                print(f"Moved {os.path.basename(lf)} to old_checkpoints/")
        
        # Also remove latest.pth and other checkpoint files that might cause auto-resuming
        checkpoint_files_to_remove = ['latest.pth', 'last_checkpoint']
        for filename in checkpoint_files_to_remove:
            filepath = os.path.join(work_dir, filename)
            if os.path.exists(filepath):
                if os.path.islink(filepath):
                    os.unlink(filepath)
                    print(f"Removed {filename} symlink")
                else:
                    backup_path = os.path.join(backup_dir, filename)
                    if not os.path.exists(backup_path):
                        shutil.move(filepath, backup_path)
                        print(f"Moved {filename} to old_checkpoints/")
    
    # Don't auto-resume to avoid the optimizer mismatch error
    if args.resume:
        print("Resume requested, but starting fresh due to architecture changes.")
        print("Old checkpoints backed up in work_dirs/kitti_3class_adversarial_training/old_checkpoints/")
    
    # Add config options - updated for 20 epochs with aggressive adversarial training
    cfg_options = [
        'train_dataloader.batch_size=6',
        'val_dataloader.batch_size=1',
        # Override the conflicting train_cfg from base config
        'train_cfg._delete_=True',  # Delete the base config's train_cfg
        'train_cfg.type=EpochBasedTrainLoop',
        'train_cfg.max_epochs=20',  # REDUCED: 20 epochs to prevent over-adaptation
        'train_cfg.val_interval=5',
        # Save checkpoint every 5 epochs
        'default_hooks.checkpoint.interval=5',
        'default_hooks.checkpoint.max_keep_ckpts=10',
        # Enable detailed logging
        'default_hooks.logger.interval=50',
        'log_level=INFO'
    ]
    
    # If freeze-detector flag is set, freeze all detector components
    if args.freeze_detector:
        print("\n" + "="*80)
        print("FROZEN DETECTOR MODE ACTIVATED")
        print("="*80)
        print("Freezing all detector components (lr_mult=0.0)")
        print("Only adversary will be trained")
        print("="*80)
        
        freeze_options = [
            # Freeze ALL detector components
            'optim_wrapper.paramwise_cfg.custom_keys.backbone.lr_mult=0.0',
            'optim_wrapper.paramwise_cfg.custom_keys.neck.lr_mult=0.0',
            'optim_wrapper.paramwise_cfg.custom_keys.bbox_head.lr_mult=0.0',
            'optim_wrapper.paramwise_cfg.custom_keys.voxel_encoder.lr_mult=0.0',
            'optim_wrapper.paramwise_cfg.custom_keys.middle_encoder.lr_mult=0.0',
            # Ensure adversary has high LR
            'optim_wrapper.paramwise_cfg.custom_keys.adversary.lr_mult=20.0',
            # More aggressive adversarial settings for frozen detector
            'model.adversarial_loss_weight=0.5',
            'model.regularization_weight=0.00001',
        ]
        cfg_options.extend(freeze_options)
    
    if cfg_options:
        train_args.extend(['--cfg-options'] + cfg_options)
    
    print(f"\nConfig: {config}")
    print(f"Work dir: {work_dir}")
    print(f"Validate: {args.validate}")
    print(f"Resume: False (starting fresh due to architecture changes)")
    print(f"Training will run for 40 epochs")  # UPDATED
    print(f"Training command: train.py {' '.join(train_args)}")
    
    # Ensure CUDA stays enabled throughout training
    print("\nEnsuring CUDA remains enabled for IoU calculations...")
    if 'NUMBA_DISABLE_CUDA' in os.environ:
        del os.environ['NUMBA_DISABLE_CUDA']
    
    # Run training
    original_argv = sys.argv.copy()
    sys.argv = ['train.py'] + train_args
    
    try:
        from mmdetection3d.tools.train import main as train_main
        print("\nStarting training...")
        train_main()
        print("\nTraining completed successfully!")
        print("\n" + "=" * 80)
        print("NEXT STEPS:")
        print("=" * 80)
        print("1. Run evaluation: python evaluate_kitti_3class.py --mode both")
        print("2. Analyze perturbation statistics in:")
        print("   - work_dirs/kitti_3class_adversarial_training/l2_norms.csv")
        print("   - work_dirs/kitti_3class_adversarial_visualizations/")
        print("=" * 80)
        return True
    except OverflowError as e:
        print(f"\nOverflowError detected: {e}")
        print("\nThis error occurs when the optimizer's step count becomes too large.")
        print("This typically happens when resuming training after many epochs.")
        print("\nPossible solutions:")
        print("1. Start fresh training without --resume flag")
        print("2. Manually reset the optimizer state in the checkpoint")
        print("3. Use a different optimizer configuration")
        
        # Try to provide checkpoint info
        import glob
        checkpoints = glob.glob(os.path.join(work_dir, '*.pth'))
        if checkpoints:
            latest = max(checkpoints, key=os.path.getctime)
            print(f"\nLatest checkpoint: {os.path.basename(latest)}")
            
        return False
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        sys.argv = original_argv

if __name__ == '__main__':
    success = main()
    if not success:
        sys.exit(1)