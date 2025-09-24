"""
Evaluation script for NuScenes adversarial implementation.
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

# Import custom hooks to register them
try:
    import custom_hook
    print("Custom hooks registered successfully")
except Exception as e:
    print(f"Warning: Could not import custom hooks: {e}")

# Apply the same path duplication fix as in the baseline evaluation
import mmengine.fileio.backends.local_backend
import os

# Override the local backend's get method to fix all path issues
original_local_get = mmengine.fileio.backends.local_backend.LocalBackend.get

def fixed_local_get(self, filepath):
    """Fixed get method that handles duplicate paths."""
    if isinstance(filepath, str):
        # Fix samples duplication
        if 'samples/LIDAR_TOP/samples/LIDAR_TOP' in filepath:
            # Remove the duplicate part
            filepath = filepath.replace('samples/LIDAR_TOP/samples/LIDAR_TOP', 'samples/LIDAR_TOP')
        
        # Fix sweeps duplication  
        elif 'sweeps/LIDAR_TOP/sweeps/LIDAR_TOP' in filepath:
            filepath = filepath.replace('sweeps/LIDAR_TOP/sweeps/LIDAR_TOP', 'sweeps/LIDAR_TOP')
        
        # Fix any remaining Windows path separator issues
        filepath = filepath.replace('\\', '/')
        
        # Also handle cases where only part of the path is duplicated
        if filepath.count('samples/LIDAR_TOP') > 1:
            parts = filepath.split('samples/LIDAR_TOP')
            # Keep only the last occurrence with proper prefix
            filepath = 'data/nuscenes/samples/LIDAR_TOP' + parts[-1]
        elif filepath.count('sweeps/LIDAR_TOP') > 1:
            parts = filepath.split('sweeps/LIDAR_TOP')
            filepath = 'data/nuscenes/sweeps/LIDAR_TOP' + parts[-1]
    
    return original_local_get(self, filepath)

# Apply the fix to the LocalBackend class
mmengine.fileio.backends.local_backend.LocalBackend.get = fixed_local_get
print("Applied comprehensive path duplication fix for all file loading")

def run_baseline_evaluation():
    """Test the baseline CenterPoint model performance."""
    print("=" * 80)
    print("TESTING BASELINE CENTERPOINT MODEL")
    print("=" * 80)
    
    # Use the original data location with simplified config
    print("Using data at: data/nuscenes/")
    
    # Use the no-sweeps baseline evaluation config to avoid path issues
    config = 'configs/centerpoint_baseline_eval_nosweeps.py'
    checkpoint = 'checkpoints/centerpoint_01voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus_20220810_052355-a6928835.pth'
    work_dir = 'work_dirs/nuscenes_baseline_01voxel_nosweeps'
    
    print(f"Config: {config}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Work dir: {work_dir}")
    
    test_args = [
        config,
        checkpoint,
        '--work-dir', work_dir
    ]
    
    # Run evaluation using the original method to see error details
    import sys
    original_argv = sys.argv.copy()
    sys.argv = ['test.py'] + test_args
    
    try:
        from mmdetection3d.tools.test import main as test_main
        test_main()
        print("\\nBaseline evaluation completed!")
        return True
    except Exception as e:
        print(f"Baseline evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        sys.argv = original_argv

def create_nuscenes_symlink():
    """Create a proper data structure to avoid path duplication."""
    import os
    import shutil
    
    # The issue is that mmdetection3d appends data_prefix to paths from pkl files
    # So we need to put files directly in data_root without the prefix structure
    target_dir = 'data/nuscenes_eval'
    source_base = 'data/nuscenes'
    
    # Create the target directory structure
    os.makedirs(target_dir, exist_ok=True)
    
    # Create direct links to LIDAR_TOP directories without the prefix structure
    samples_source = os.path.join(source_base, 'samples', 'LIDAR_TOP')
    sweeps_source = os.path.join(source_base, 'sweeps', 'LIDAR_TOP') 
    
    # Create direct symlinks so paths work correctly
    if os.path.exists(samples_source):
        samples_target = os.path.join(target_dir, 'samples')
        os.makedirs(samples_target, exist_ok=True)
        samples_lidar_target = os.path.join(samples_target, 'LIDAR_TOP')
        
        if not os.path.exists(samples_lidar_target):
            try:
                os.symlink(os.path.abspath(samples_source), samples_lidar_target, target_is_directory=True)
                print(f"Created samples symlink: {samples_lidar_target}")
            except OSError as e:
                print(f"Failed to create samples symlink: {e}")
    
    if os.path.exists(sweeps_source):
        sweeps_target = os.path.join(target_dir, 'sweeps')
        os.makedirs(sweeps_target, exist_ok=True)
        sweeps_lidar_target = os.path.join(sweeps_target, 'LIDAR_TOP')
        
        if not os.path.exists(sweeps_lidar_target):
            try:
                os.symlink(os.path.abspath(sweeps_source), sweeps_lidar_target, target_is_directory=True)
                print(f"Created sweeps symlink: {sweeps_lidar_target}")
            except OSError as e:
                print(f"Failed to create sweeps symlink: {e}")
    
    # Copy other necessary files
    other_dirs = ['v1.0-trainval', 'maps']
    for dir_name in other_dirs:
        source_path = os.path.join(source_base, dir_name)
        target_path = os.path.join(target_dir, dir_name)
        if os.path.exists(source_path) and not os.path.exists(target_path):
            try:
                os.symlink(os.path.abspath(source_path), target_path, target_is_directory=True)
                print(f"Created symlink: {target_path}")
            except OSError as e:
                print(f"Failed to create symlink for {dir_name}: {e}")
    
    # Copy pkl files
    pkl_files = ['nuscenes_infos_val.pkl', 'nuscenes_infos_train.pkl']
    for pkl_file in pkl_files:
        source_path = os.path.join(source_base, pkl_file)
        target_path = os.path.join(target_dir, pkl_file)
        if os.path.exists(source_path) and not os.path.exists(target_path):
            try:
                os.symlink(os.path.abspath(source_path), target_path)
                print(f"Created pkl symlink: {target_path}")
            except OSError as e:
                print(f"Failed to create pkl symlink: {e}")
    
    return target_dir

def run_adversarial_evaluation(adversarial_mode=True):
    """Test the adversarial model performance."""
    mode_str = "ADVERSARIAL" if adversarial_mode else "CLEAN"
    print("=" * 80)
    print(f"TESTING {mode_str} MODE WITH FIXED ADVERSARIAL MODEL")
    print("=" * 80)
    
    # Use the adversarial configuration
    config = 'configs/adversarial/adversarial-centerpoint_voxel-nuscenes.py'
    checkpoint = 'checkpoints/centerpoint_01voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus_20220810_052355-a6928835.pth'
    work_dir = f'work_dirs/nuscenes_adversarial_fixed_{mode_str.lower()}'
    
    test_args = [
        config,
        checkpoint,
        '--work-dir', work_dir
    ]
    
    # Configure for clean vs adversarial evaluation
    cfg_options = []
    
    if not adversarial_mode:
        # Disable adversarial components for clean evaluation
        cfg_options.extend([
            'model.adversary_cfg=None',
            'model.adversarial_loss_weight=0.0',
            'model.regularization_weight=0.0',
            'custom_hooks=[]'
        ])
        print("Running in CLEAN mode (no perturbations)")
    else:
        print("Running in ADVERSARIAL mode (with perturbations)")
    
    if cfg_options:
        test_args.extend(['--cfg-options'] + cfg_options)
    
    print(f"Config: {config}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Work dir: {work_dir}")
    
    # Run evaluation
    import sys
    original_argv = sys.argv.copy()
    sys.argv = ['test.py'] + test_args
    
    try:
        from mmdetection3d.tools.test import main as test_main
        test_main()
        print(f"\\n{mode_str} evaluation completed!")
        return True
    except Exception as e:
        print(f"{mode_str} evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        sys.argv = original_argv

def main():
    parser = argparse.ArgumentParser(description='Evaluate fixed NuScenes adversarial implementation')
    parser.add_argument('--mode', choices=['baseline', 'clean', 'adversarial', 'all'], 
                       default='all', help='Evaluation mode')
    args = parser.parse_args()
    
    results = {}
    
    if args.mode in ['baseline', 'all']:
        print("\\nStep 1: Testing baseline CenterPoint model...")
        results['baseline'] = run_baseline_evaluation()
    
    if args.mode in ['clean', 'all']:
        print("\\nStep 2: Testing adversarial model in clean mode...")
        results['clean'] = run_adversarial_evaluation(adversarial_mode=False)
    
    if args.mode in ['adversarial', 'all']:
        print("\\nStep 3: Testing adversarial model with perturbations...")
        results['adversarial'] = run_adversarial_evaluation(adversarial_mode=True)
    
    # Summary
    print("\\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    for mode, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{mode.upper():12} : {status}")
    
    print("\\nNext steps:")
    print("1. Check the work_dirs for detailed results")
    print("2. Compare mAP values between baseline, clean, and adversarial modes")
    print("3. If adversarial mode shows similar performance to clean mode, increase adversarial_loss_weight")
    print("4. If baseline fails, check data paths and model compatibility")

if __name__ == '__main__':
    main()