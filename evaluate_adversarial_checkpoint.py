"""
Simple script to evaluate the latest adversarial checkpoint.
This script evaluates a specific checkpoint from adversarial training.
"""
import os
import sys
import glob
import json
import argparse
from datetime import datetime

# Force use of GPU 1 (second GPU) to avoid conflicts
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Apply all necessary fixes
try:
    import fix_numpy_compat
    print("Numpy compatibility fixes applied")
except:
    pass

import mmengine.config
def _fake_pretty_text(self):
    return str(self._cfg_dict)
mmengine.config.Config.pretty_text = property(_fake_pretty_text)

# Apply path duplication fix
import mmengine.fileio.backends.local_backend
original_local_get = mmengine.fileio.backends.local_backend.LocalBackend.get

def fixed_local_get(self, filepath):
    """Fixed get method that handles duplicate paths."""
    if isinstance(filepath, str):
        filepath = filepath.replace('\\', '/')
        if 'data/nuscenes/data/nuscenes' in filepath:
            filepath = filepath.replace('data/nuscenes/data/nuscenes', 'data/nuscenes')
        if 'samples/LIDAR_TOP/samples/LIDAR_TOP' in filepath:
            filepath = filepath.replace('samples/LIDAR_TOP/samples/LIDAR_TOP', 'samples/LIDAR_TOP')
        if 'sweeps/LIDAR_TOP/sweeps/LIDAR_TOP' in filepath:
            filepath = filepath.replace('sweeps/LIDAR_TOP/sweeps/LIDAR_TOP', 'sweeps/LIDAR_TOP')
        if filepath.startswith('../data/nuscenes'):
            filepath = filepath.replace('../data/nuscenes', 'data/nuscenes')
        if filepath.count('samples/LIDAR_TOP') > 1:
            parts = filepath.split('samples/LIDAR_TOP')
            filepath = 'data/nuscenes/samples/LIDAR_TOP' + parts[-1]
        elif filepath.count('sweeps/LIDAR_TOP') > 1:
            parts = filepath.split('sweeps/LIDAR_TOP')
            filepath = 'data/nuscenes/sweeps/LIDAR_TOP' + parts[-1]
    return original_local_get(self, filepath)

mmengine.fileio.backends.local_backend.LocalBackend.get = fixed_local_get
print("Applied path duplication fix")

# Apply dbsampler fix
try:
    from fix_dbsampler_indexing import patch_dbsampler
    patch_dbsampler()
except:
    pass

# Import custom hooks
try:
    import custom_hook
    print("Custom hooks registered")
except:
    pass

def main():
    parser = argparse.ArgumentParser(description='Evaluate adversarial checkpoint')
    parser.add_argument('--checkpoint', type=str, help='Path to specific checkpoint file')
    args = parser.parse_args()
    
    if args.checkpoint:
        # Use specified checkpoint
        latest_checkpoint = args.checkpoint
        if not os.path.exists(latest_checkpoint):
            print(f"Checkpoint not found: {latest_checkpoint}")
            return
        # Extract epoch number from filename
        basename = os.path.basename(latest_checkpoint)
        if 'epoch_' in basename:
            latest_epoch = int(basename.split('epoch_')[1].split('.')[0])
        else:
            latest_epoch = "unknown"
    else:
        # Find latest adversarial checkpoint
        checkpoint_pattern = 'work_dirs/nuscenes_adversarial_training_fixed/epoch_*.pth'
        checkpoints = glob.glob(checkpoint_pattern)
        
        if not checkpoints:
            print("No adversarial checkpoints found!")
            return
        
        # Sort by epoch number
        epochs = [(int(os.path.basename(cp).split('_')[1].split('.')[0]), cp) for cp in checkpoints]
        epochs.sort()
        latest_epoch, latest_checkpoint = epochs[-1]
    
    print(f"\n{'='*60}")
    print(f"EVALUATING ADVERSARIAL CHECKPOINT")
    print(f"{'='*60}")
    print(f"Checkpoint: {latest_checkpoint}")
    print(f"Epoch: {latest_epoch}")
    print(f"File size: {os.path.getsize(latest_checkpoint)/(1024**3):.2f} GB")
    
    # Configuration
    config = 'configs/adversarial/adversarial-centerpoint_voxel-nuscenes-fixed.py'
    work_dir = f'work_dirs/adversarial_epoch{latest_epoch}_evaluation'
    
    # Create work directory
    os.makedirs(work_dir, exist_ok=True)
    
    # Prepare test arguments
    test_args = [
        config,
        latest_checkpoint,
        '--work-dir', work_dir
    ]
    
    print(f"\nRunning evaluation...")
    print(f"Config: {config}")
    print(f"Work dir: {work_dir}")
    
    # Run evaluation
    original_argv = sys.argv.copy()
    sys.argv = ['test.py'] + test_args
    
    try:
        from mmdetection3d.tools.test import main as test_main
        test_main()
        print(f"\n✅ Evaluation completed successfully!")
        print(f"Results saved to: {work_dir}")
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        sys.argv = original_argv
    
    print("\n" + "="*60)

if __name__ == '__main__':
    main()