"""
Evaluation script for KITTI 3-class dataset with proper configuration handling.
"""
import argparse
import os
import sys

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Fix numpy compatibility
try:
    import fix_numpy_compat
    print("Numpy compatibility fixes applied")
except Exception as e:
    print(f"Warning: Could not apply numpy compatibility fixes: {e}")

# Fix yapf compatibility issue
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

# Import custom hooks to register them
try:
    import custom_hook
    print("Custom hooks registered successfully")
except Exception as e:
    print(f"Warning: Could not import custom hooks: {e}")

def run_baseline_evaluation():
    """Test the baseline SECOND model performance on 3-class KITTI."""
    print("=" * 80)
    print("TESTING BASELINE SECOND MODEL (3-CLASS)")
    print("=" * 80)
    
    # Enable CUDA for evaluation (needed for IoU calculations)
    import os
    if 'NUMBA_DISABLE_CUDA' in os.environ:
        del os.environ['NUMBA_DISABLE_CUDA']
        print("Enabled CUDA for evaluation metrics calculation")
    
    # Use the 3-class config with the downloaded checkpoint
    config = 'mmdetection3d/configs/second/second_hv_secfpn_8xb6-80e_kitti-3d-3class.py'
    # Try fixed checkpoint first, fallback to original
    checkpoint = 'checkpoints/second_hv_secfpn_8xb6-80e_kitti-3d-3class-fixed.pth'
    if not os.path.exists(checkpoint):
        checkpoint = 'checkpoints/second_hv_secfpn_8xb6-80e_kitti-3d-3class-b086d0a3.pth'
    work_dir = 'work_dirs/kitti_3class_baseline_evaluation'
    
    print(f"Config: {config}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Work dir: {work_dir}")
    print("Classes: Car, Pedestrian, Cyclist")
    
    # Check if files exist
    if not os.path.exists(config):
        print(f"ERROR: Config file not found: {config}")
        print("Please ensure you have the mmdetection3d configs.")
        return False
        
    if not os.path.exists(checkpoint):
        print(f"ERROR: Checkpoint file not found: {checkpoint}")
        print("Please ensure you have downloaded the pretrained model.")
        return False
    
    # Use standard val split
    test_args = [
        config,
        checkpoint,
        '--work-dir', work_dir
    ]
    
    # Run evaluation using the same approach as NuScenes
    import sys
    original_argv = sys.argv.copy()
    sys.argv = ['test.py'] + test_args
    
    try:
        from mmdetection3d.tools.test import main as test_main
        test_main()
        print("\nBaseline 3-class evaluation completed!")
        return True
    except Exception as e:
        print(f"Baseline evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        sys.argv = original_argv

def run_adversarial_evaluation(checkpoint_path=None):
    """Test the adversarial model performance on 3-class KITTI."""
    print("=" * 80)
    print("TESTING ADVERSARIAL MODEL (3-CLASS)")
    print("=" * 80)
    
    # Enable CUDA for evaluation (needed for IoU calculations)
    import os
    if 'NUMBA_DISABLE_CUDA' in os.environ:
        del os.environ['NUMBA_DISABLE_CUDA']
        print("Enabled CUDA for evaluation metrics calculation")
    
    config = 'configs/adversarial/adversarial-second_hv_secfpn_8xb6-80e_kitti-3d-3class.py'
    
    # Auto-detect checkpoint if not provided
    if checkpoint_path is None:
        # Try to find the latest checkpoint
        training_dir = 'work_dirs/kitti_3class_adversarial_training'
        
        # Check for common checkpoint names (prioritize latest epoch_40.pth from new training)
        possible_checkpoints = ['epoch_40.pth', 'latest.pth', 'epoch_20.pth', 'epoch_80.pth']
        
        for ckpt_name in possible_checkpoints:
            ckpt_path = os.path.join(training_dir, ckpt_name)
            if os.path.exists(ckpt_path):
                checkpoint_path = ckpt_path
                print(f"Found checkpoint: {ckpt_name}")
                break
        else:
            # If no standard names found, look for any .pth file
            import glob
            pth_files = glob.glob(os.path.join(training_dir, '*.pth'))
            if pth_files:
                # Use the most recent one
                checkpoint_path = max(pth_files, key=os.path.getctime)
                print(f"Found checkpoint: {os.path.basename(checkpoint_path)}")
            else:
                print("ERROR: No checkpoint found. Please provide checkpoint path or train the model first.")
                print(f"Searched in: {training_dir}")
                return False
    
    work_dir = 'work_dirs/kitti_3class_adversarial_evaluation'
    
    print(f"Config: {config}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Work dir: {work_dir}")
    print("Classes: Car, Pedestrian, Cyclist")
    
    test_args = [
        config,
        checkpoint_path,
        '--work-dir', work_dir,
        '--cfg-options',
        'test_dataloader.dataset.data_root=data/kitti/',
        'test_dataloader.dataset.data_prefix.pts=training/velodyne_reduced',
        'test_evaluator.ann_file=data/kitti/kitti_infos_val.pkl'
    ]
    
    # Run evaluation
    import sys
    original_argv = sys.argv.copy()
    sys.argv = ['test.py'] + test_args
    
    try:
        from mmdetection3d.tools.test import main as test_main
        test_main()
        print("\nAdversarial 3-class evaluation completed!")
        return True
    except Exception as e:
        print(f"Adversarial evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        sys.argv = original_argv

def main():
    parser = argparse.ArgumentParser(description='Evaluate KITTI 3-class models')
    parser.add_argument('--mode', type=str, choices=['clean', 'adversarial', 'both'], 
                        default='both', help='Evaluation mode')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to adversarial checkpoint (auto-detect if not provided)')
    
    args = parser.parse_args()
    
    success = True
    
    if args.mode in ['clean', 'both']:
        print("\n" + "="*80)
        print("BASELINE EVALUATION (3-CLASS)")
        print("="*80)
        success = run_baseline_evaluation() and success
    
    if args.mode in ['adversarial', 'both']:
        print("\n" + "="*80)
        print("ADVERSARIAL EVALUATION (3-CLASS)")
        print("="*80)
        success = run_adversarial_evaluation(args.checkpoint) and success
    
    if success:
        print("\n" + "="*80)
        print("ALL EVALUATIONS COMPLETED SUCCESSFULLY!")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("SOME EVALUATIONS FAILED!")
        print("="*80)
        sys.exit(1)

if __name__ == '__main__':
    main()