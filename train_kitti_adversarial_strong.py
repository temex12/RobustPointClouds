#!/usr/bin/env python
"""
Train KITTI with strong adversarial attack.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Fix numpy compatibility
try:
    import fix_numpy_compat
    print("Numpy compatibility fixes applied")
except Exception as e:
    print(f"Warning: Could not apply numpy compatibility fixes: {e}")

# Ensure CUDA is enabled
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

# Import custom hooks
try:
    import custom_hook
    print("Custom hooks registered successfully")
except Exception as e:
    print(f"Warning: Could not import custom hooks: {e}")

# Import strong adversarial modules to ensure registration
try:
    from models.detectors.strong_adversarial_voxelnet import StrongAdversarialVoxelNet
    from models.adversarial.strong_voxel_perturber import StrongVoxelPerturber
    print("Strong adversarial modules imported and registered successfully")
except Exception as e:
    print(f"Warning: Could not import enhanced modules: {e}")

# Fix adversary optimizer issue
try:
    from fix_adversary_optimizer import fix_adversary_optimizer_checkpoint
except Exception as e:
    print(f"Warning: Could not import adversary optimizer fix: {e}")

# Enhanced configuration
CONFIG = 'configs/adversarial/adversarial-second_stronger_enhanced.py'
WORK_DIR = 'work_dirs/kitti_enhanced_stronger_adversarial'

class AdversarialProgressMonitor:
    """Monitor adversarial training progress and dynamics."""
    
    def __init__(self):
        self.adversarial_losses = []
        self.detection_losses = []
        self.scaling_factors = []
        self.epochs = []
        self.l2_norms = []
        
    def parse_log_line(self, line):
        """Parse training log line to extract adversarial metrics."""
        if 'loss_adversarial:' in line:
            # Extract adversarial loss
            import re
            adv_match = re.search(r'loss_adversarial:\s*([-\d\.]+)', line)
            det_match = re.search(r'loss_cls:\s*([\d\.]+)', line)
            epoch_match = re.search(r'Epoch\(train\)\s*\[(\d+)\]', line)
            
            if adv_match and det_match and epoch_match:
                adv_loss = float(adv_match.group(1))
                det_loss = float(det_match.group(1))
                epoch = int(epoch_match.group(1))
                
                self.adversarial_losses.append(adv_loss)
                self.detection_losses.append(det_loss)
                self.epochs.append(epoch)
                
                return True
        return False
    
    def plot_progress(self, save_path=None):
        """Plot adversarial training progress."""
        if not self.adversarial_losses:
            print("No adversarial loss data to plot")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Adversarial loss magnitude over time
        ax1.plot(self.adversarial_losses, 'r-', label='Adversarial Loss', linewidth=2)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Adversarial Loss')
        ax1.set_title('Adversarial Loss Progression')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Detection loss progression
        ax2.plot(self.detection_losses, 'b-', label='Detection Loss', linewidth=2)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Detection Loss')
        ax2.set_title('Detection Loss Progression')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Plot 3: Adversarial loss strength over epochs
        if len(set(self.epochs)) > 1:
            epoch_adv_strength = {}
            for epoch, adv_loss in zip(self.epochs, self.adversarial_losses):
                if epoch not in epoch_adv_strength:
                    epoch_adv_strength[epoch] = []
                epoch_adv_strength[epoch].append(abs(adv_loss))
            
            epochs = sorted(epoch_adv_strength.keys())
            avg_strength = [np.mean(epoch_adv_strength[e]) for e in epochs]
            max_strength = [np.max(epoch_adv_strength[e]) for e in epochs]
            
            ax3.plot(epochs, avg_strength, 'g-o', label='Average Strength', linewidth=2)
            ax3.plot(epochs, max_strength, 'r--o', label='Max Strength', linewidth=2)
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Adversarial Loss Magnitude')
            ax3.set_title('Adversarial Strength by Epoch')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
        
        # Plot 4: Success indicator
        recent_strength = np.mean([abs(x) for x in self.adversarial_losses[-50:]]) if len(self.adversarial_losses) >= 50 else 0
        initial_strength = np.mean([abs(x) for x in self.adversarial_losses[:50]]) if len(self.adversarial_losses) >= 50 else 0
        
        if initial_strength > 0:
            strength_ratio = recent_strength / initial_strength
            success_indicator = "SUCCESSFUL" if strength_ratio >= 0.8 else "NEEDS_ADJUSTMENT"
            color = 'green' if success_indicator == "SUCCESSFUL" else 'red'
        else:
            strength_ratio = 0
            success_indicator = "INSUFFICIENT_DATA"
            color = 'orange'
        
        ax4.text(0.5, 0.7, f"Adversarial Training Status:", ha='center', va='center', 
                transform=ax4.transAxes, fontsize=14, fontweight='bold')
        ax4.text(0.5, 0.5, success_indicator, ha='center', va='center', 
                transform=ax4.transAxes, fontsize=16, fontweight='bold', color=color)
        ax4.text(0.5, 0.3, f"Strength Retention: {strength_ratio:.2f}", ha='center', va='center', 
                transform=ax4.transAxes, fontsize=12)
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Progress plot saved to: {save_path}")
        
        plt.show()

def main():
    # Check if config exists
    if not os.path.exists(CONFIG):
        print(f"Error: Config file not found: {CONFIG}")
        sys.exit(1)
    
    print("=" * 80)
    print("🚀 KITTI ENHANCED STRONGER ADVERSARIAL TRAINING")
    print("=" * 80)
    print(f"Config: {CONFIG}")
    print(f"Work dir: {WORK_DIR}")
    print()
    print("🔬 ENHANCED FEATURES:")
    print("  ✅ Dynamic adversarial scaling (strength increases over epochs)")
    print("  ✅ Curriculum learning (gradual complexity increase)")
    print("  ✅ Momentum-based adversarial updates")
    print("  ✅ Anti-adaptation mechanisms (prevent detector over-adaptation)")
    print("  ✅ Enhanced monitoring and visualization")
    print()
    print("🎯 EXPECTED IMPROVEMENTS:")
    print("  • Adversarial attacks maintain/increase strength over epochs")
    print("  • More effective perturbations that don't weaken")
    print("  • Better balance between detector and adversary learning")
    print("  • Sustained 5-25% performance drops throughout training")
    print("=" * 80)
    
    # Store original argv
    original_argv = sys.argv.copy()
    
    # Initialize progress monitor
    monitor = AdversarialProgressMonitor()
    
    try:
        # Import mmdetection3d train function
        from mmdetection3d.tools.train import main as train_main
        
        # Set up arguments for training
        sys.argv = [
            'train.py',
            CONFIG,
            '--work-dir', WORK_DIR
        ]
        
        # Add enhanced config options
        cfg_options = [
            'train_dataloader.batch_size=1',      # Batch size 1 to avoid tensor collation issues
            'train_dataloader.num_workers=0',     # Disable multiprocessing
            'val_dataloader.batch_size=1',
            'val_dataloader.num_workers=0',
            'train_cfg._delete_=True',
            'train_cfg.type=EpochBasedTrainLoop',
            'train_cfg.max_epochs=10',            # Longer training for scaling effects
            'train_cfg.val_interval=2',
            'default_hooks.checkpoint.interval=2',
            'default_hooks.checkpoint.max_keep_ckpts=8',
            'default_hooks.logger.interval=25',
            'log_level=INFO'
        ]
        
        # Add cfg-options
        for opt in cfg_options:
            sys.argv.extend(['--cfg-options', opt])
        
        print("\\nStarting enhanced adversarial training...")
        print("Monitor the logs for:")
        print("  🔥 [ENHANCED PERTURBATION] - Dynamic scaling information")
        print("  ⚡ [DYNAMIC SCALING] - Automatic strength adjustments")
        print("  🛡️ [ANTI-ADAPTATION] - Detector adaptation prevention")
        print()
        
        # Call the training function
        train_main()
        
        print("\\n" + "="*80)
        print("✅ ENHANCED ADVERSARIAL TRAINING COMPLETED!")
        print("="*80)
        
        # Parse log file for analysis
        log_files = [f for f in os.listdir(WORK_DIR) if f.endswith('.log')]
        if log_files:
            latest_log = os.path.join(WORK_DIR, sorted(log_files)[-1])
            print(f"Parsing log file: {latest_log}")
            
            with open(latest_log, 'r') as f:
                for line in f:
                    monitor.parse_log_line(line)
            
            # Generate progress plot
            plot_path = os.path.join(WORK_DIR, 'adversarial_training_progress.png')
            monitor.plot_progress(plot_path)
        
        print("\\n🎯 NEXT STEPS:")
        print("  1. Run evaluation to measure final performance drops")
        print("  2. Compare with baseline results")
        print("  3. Analyze adversarial strength progression")
        print("  4. Generate comprehensive analysis notebook")
        
        return True
        
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