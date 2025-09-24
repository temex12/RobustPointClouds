# Handle import errors gracefully
try:
    from .voxel_perturber import VoxelPerturber
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import VoxelPerturber: {e}. Using fallback implementation.")
    # Try fallback import
    try:
        from .voxel_perturber_safe import VoxelPerturber
    except ImportError:
        # If fallback also fails, create a dummy class
        class VoxelPerturber:
            def __init__(self, *args, **kwargs):
                raise ImportError("VoxelPerturber could not be imported. Please check your installation.")

# Enhanced VoxelPerturber with dynamic scaling
try:
    from .enhanced_voxel_perturber import EnhancedVoxelPerturber
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import EnhancedVoxelPerturber: {e}")
    EnhancedVoxelPerturber = None

try:
    from .visualization import AdversarialVisualizer, visualize_voxel_perturbations
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import visualization tools: {e}")
    AdversarialVisualizer = None
    visualize_voxel_perturbations = None

__all__ = ['VoxelPerturber', 'EnhancedVoxelPerturber', 'AdversarialVisualizer', 'visualize_voxel_perturbations']