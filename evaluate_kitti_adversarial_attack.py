"""
Evaluate KITTI model under adversarial attack by forcing perturbations during evaluation.
"""
import argparse
import os
import sys

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Apply necessary fixes
try:
    import fix_numpy_compat
    print("Numpy compatibility fixes applied")
except Exception as e:
    print(f"Warning: Could not apply numpy compatibility fixes: {e}")

import mmengine.config
def _fake_pretty_text(self):
    return str(self._cfg_dict)
mmengine.config.Config.pretty_text = property(_fake_pretty_text)

def patch_model_for_adversarial_eval():
    """Monkey patch to force perturbations during evaluation."""
    from models.detectors.adversarial_voxelnet import AdversarialVoxelNet
    
    # Save original extract_feat method
    original_extract_feat = AdversarialVoxelNet.extract_feat
    
    def patched_extract_feat(self, batch_inputs_dict):
        """Modified extract_feat that applies perturbations even in eval mode."""
        # Force the adversarial_disabled flag to False to ensure perturbations
        original_training = self.training
        original_disabled = getattr(self, '_adversarial_disabled', False)
        
        # Enable training mode and adversarial perturbations
        self.training = True
        self._adversarial_disabled = False
        
        # Debug: Confirm we have adversary and it's enabled
        has_adversary = hasattr(self, 'adversary') and self.adversary is not None
        print(f"[PATCH DEBUG] training={self.training}, disabled={self._adversarial_disabled}, has_adversary={has_adversary}")
        
        # Call original method with perturbations enabled
        result = original_extract_feat(self, batch_inputs_dict)
        
        # Check if perturbations were actually applied
        if hasattr(self, '_current_l2_norm') and self._current_l2_norm is not None:
            l2_val = self._current_l2_norm.item()
            print(f"[PATCH SUCCESS] L2 norm applied: {l2_val:.6f}")
        else:
            print(f"[PATCH FAILURE] No L2 norm found - perturbations not applied!")
        
        # Restore original states
        self.training = original_training
        self._adversarial_disabled = original_disabled
        
        return result
    
    # Apply patch
    AdversarialVoxelNet.extract_feat = patched_extract_feat
    print("[Adversarial Eval] Patched model to apply perturbations during evaluation")

def run_adversarial_attack_evaluation(checkpoint_path=None):
    """Test model performance under adversarial attack."""
    print("=" * 80)
    print("TESTING MODEL UNDER ADVERSARIAL ATTACK")
    print("=" * 80)
    print("This evaluation applies perturbations during testing")
    print("to measure robustness against adversarial attacks")
    print("=" * 80)
    
    # Apply patch to force perturbations
    patch_model_for_adversarial_eval()
    
    # Enable CUDA
    if 'NUMBA_DISABLE_CUDA' in os.environ:
        del os.environ['NUMBA_DISABLE_CUDA']
    
    # Use fixed config if available, otherwise original
    config_paths = [
        'configs/adversarial/adversarial-second_fixed.py',
        'configs/adversarial/adversarial-second_hv_secfpn_8xb6-80e_kitti-3d-3class.py'
    ]
    
    config = None
    for cfg_path in config_paths:
        if os.path.exists(cfg_path):
            config = cfg_path
            break
    
    if config is None:
        print("ERROR: No adversarial config found")
        return False
    
    # Auto-detect checkpoint
    if checkpoint_path is None:
        training_dirs = [
            'work_dirs/kitti_3class_adversarial_fixed',
            'work_dirs/kitti_3class_adversarial_training'
        ]
        
        for training_dir in training_dirs:
            if not os.path.exists(training_dir):
                continue
                
            possible_checkpoints = ['epoch_20.pth', 'latest.pth', 'epoch_40.pth']
            
            for ckpt_name in possible_checkpoints:
                ckpt_path = os.path.join(training_dir, ckpt_name)
                if os.path.exists(ckpt_path):
                    checkpoint_path = ckpt_path
                    print(f"Found checkpoint: {ckpt_path}")
                    break
            
            if checkpoint_path:
                break
        
        if checkpoint_path is None:
            print("ERROR: No checkpoint found")
            return False
    
    work_dir = 'work_dirs/kitti_3class_adversarial_attack_eval'
    
    print(f"Config: {config}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Work dir: {work_dir}")
    print("\nNOTE: Perturbations WILL be applied during this evaluation")
    
    test_args = [
        config,
        checkpoint_path,
        '--work-dir', work_dir
    ]
    
    # Run evaluation
    original_argv = sys.argv.copy()
    sys.argv = ['test.py'] + test_args
    
    try:
        from mmdetection3d.tools.test import main as test_main
        test_main()
        print("\nAdversarial attack evaluation completed!")
        return True
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        sys.argv = original_argv

def main():
    parser = argparse.ArgumentParser(description='Evaluate KITTI model under adversarial attack')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint (auto-detect if not provided)')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("ADVERSARIAL ATTACK EVALUATION")
    print("="*80)
    print("This script forces perturbations during evaluation to test robustness")
    print("Compare results with clean evaluation to measure attack effectiveness")
    print("="*80)
    
    success = run_adversarial_attack_evaluation(args.checkpoint)
    
    if success:
        print("\n" + "="*80)
        print("EVALUATION COMPLETED!")
        print("="*80)
        print("\nNext steps:")
        print("1. Compare these results with baseline evaluation")
        print("2. Check L2 regularization if results are unexpected")
    else:
        sys.exit(1)

if __name__ == '__main__':
    main()