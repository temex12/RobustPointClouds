#from mmcv.utils.registry import Registry
from mmengine.registry import Registry

# Initialize a registry for adversarial models; this will be used to register any module of type 'adversary'
ADVERSARIES = Registry('adversaries')

def build_adversary(cfg):
    """Build adversary."""
    # This builds an object from the adversarial registry based on the configuration provided.
    return ADVERSARIES.build(cfg)
