"""
Training script for NuScenes adversarial implementation.
"""
import argparse
import os
import sys

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Fix compatibility issues
try:
    import fix_numpy_compat
    print("Numpy compatibility fixes applied")
except Exception as e:
    print(f"Warning: Could not apply numpy compatibility fixes: {e}")

try:
    from fix_yapf_issue import patch_mmengine_config
    patch_mmengine_config()
    print("Successfully patched mmengine Config.pretty_text")
except Exception as e:
    print(f"Warning: Could not apply yapf patch: {e}")

# More robust yapf fix - completely override the pretty_text property
import mmengine.config

def _fake_pretty_text(self):
    """Return config as string without yapf formatting."""
    return str(self._cfg_dict)

# Override the property
mmengine.config.Config.pretty_text = property(_fake_pretty_text)
print("Successfully overridden Config.pretty_text to bypass yapf")

# Apply the same path duplication fix as in the baseline evaluation
import mmengine.fileio.backends.local_backend
import os

# Override the local backend's get method to fix all path issues
original_local_get = mmengine.fileio.backends.local_backend.LocalBackend.get

def fixed_local_get(self, filepath):
    """Fixed get method that handles duplicate paths."""
    if isinstance(filepath, str):
        # First normalize all path separators to forward slashes
        filepath = filepath.replace('\\', '/')
        
        # Fix the main data/nuscenes duplication issue
        if 'data/nuscenes/data/nuscenes' in filepath:
            filepath = filepath.replace('data/nuscenes/data/nuscenes', 'data/nuscenes')
        
        # Fix samples duplication
        if 'samples/LIDAR_TOP/samples/LIDAR_TOP' in filepath:
            filepath = filepath.replace('samples/LIDAR_TOP/samples/LIDAR_TOP', 'samples/LIDAR_TOP')
        
        # Fix sweeps duplication  
        if 'sweeps/LIDAR_TOP/sweeps/LIDAR_TOP' in filepath:
            filepath = filepath.replace('sweeps/LIDAR_TOP/sweeps/LIDAR_TOP', 'sweeps/LIDAR_TOP')
        
        # Handle relative path issues
        if filepath.startswith('../data/nuscenes'):
            filepath = filepath.replace('../data/nuscenes', 'data/nuscenes')
        
        # Also handle cases where only part of the path is duplicated
        if filepath.count('samples/LIDAR_TOP') > 1:
            parts = filepath.split('samples/LIDAR_TOP')
            filepath = 'data/nuscenes/samples/LIDAR_TOP' + parts[-1]
        elif filepath.count('sweeps/LIDAR_TOP') > 1:
            parts = filepath.split('sweeps/LIDAR_TOP')
            filepath = 'data/nuscenes/sweeps/LIDAR_TOP' + parts[-1]
    
    return original_local_get(self, filepath)

# Apply the fix to the LocalBackend class
mmengine.fileio.backends.local_backend.LocalBackend.get = fixed_local_get
print("Applied comprehensive path duplication fix for all file loading")

# Apply dbsampler indexing fix
try:
    from fix_dbsampler_indexing import patch_dbsampler
    patch_dbsampler()
except Exception as e:
    print(f"Warning: Could not apply dbsampler fix: {e}")

# Import custom hooks to register them
try:
    import custom_hook
    print("Custom hooks registered successfully")
except Exception as e:
    print(f"Warning: Could not import custom hooks: {e}")

def main():
    parser = argparse.ArgumentParser(description='Train fixed NuScenes adversarial model')
    parser.add_argument('--config',
                       default='configs/adversarial/adversarial-centerpoint_voxel-nuscenes.py',
                       help='Training config file path')
    parser.add_argument('--work-dir', 
                       default='work_dirs/nuscenes_adversarial_training_fixed',
                       help='Working directory to save training results')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from latest checkpoint')
    parser.add_argument('--validate', action='store_true', default=True,
                       help='Whether to evaluate during training')
    parser.add_argument('--cfg-options', nargs='+',
                       help='Override config options')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("TRAINING FIXED NUSCENES ADVERSARIAL MODEL")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"Work dir: {args.work_dir}")
    print(f"Resume: {args.resume}")
    print(f"Validate: {args.validate}")
    
    # Check that config file exists
    if not os.path.exists(args.config):
        print(f"ERROR: Config file not found: {args.config}")
        return
    
    # Check that checkpoint exists
    checkpoint_path = 'checkpoints/centerpoint_0075voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220810_011659-04cb3a3b.pth'
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Pretrained checkpoint not found: {checkpoint_path}")
        print("Please download the CenterPoint checkpoint first.")
        return
    
    # Prepare training arguments
    train_args = [
        args.config,
        '--work-dir', args.work_dir
    ]
    
    if args.resume:
        train_args.append('--resume')
    
    if not args.validate:
        train_args.append('--no-validate')
    
    # Add config options
    cfg_options = []
    if args.cfg_options:
        cfg_options.extend(args.cfg_options)
    
    # Override some settings for safer training (paths already fixed in config)
    cfg_options.extend([
        # Conservative training settings
        'optim_wrapper.optimizer.lr=0.00005',  # Conservative learning rate
        'default_hooks.logger.interval=20',  # More frequent logging
        'default_hooks.checkpoint.interval=2'  # Save checkpoints more frequently
    ])
    
    if cfg_options:
        train_args.extend(['--cfg-options'] + cfg_options)
    
    print(f"Training arguments: {train_args}")
    
    # Run training
    import sys
    original_argv = sys.argv.copy()
    sys.argv = ['train.py'] + train_args
    
    try:
        from mmdetection3d.tools.train import main as train_main
        train_main()
        print("\\nTraining completed successfully!")
        
        # Print next steps
        print("\\n" + "=" * 80)
        print("TRAINING COMPLETED - NEXT STEPS")
        print("=" * 80)
        print(f"1. Check training logs in: {args.work_dir}")
        print("2. Run evaluation to test adversarial effectiveness:")
        print(f"   python evaluate_nuscenes_fixed.py --mode all")
        print("3. Monitor perturbation L2 norms in the logs")
        print("4. If attack is not effective, try:")
        print("   - Increase adversarial_loss_weight gradually (0.05 → 0.1 → 0.2)")
        print("   - Train for more epochs")
        print("   - Check that gradients are flowing through the adversary")
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        sys.argv = original_argv

if __name__ == '__main__':
    main()