#from mmcv.utils.registry import Registry
from mmengine.registry import Registry

# Initialize a registry for adversarial models; this will be used to register any module of type 'adversary'
# Set parent to None to avoid scope issues
ADVERSARIES = Registry('adversaries', parent=None, scope='models')

def build_adversary(cfg):
    """Build adversary."""
    # This builds an object from the adversarial registry based on the configuration provided.
    return ADVERSARIES.build(cfg)
