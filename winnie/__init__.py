import os
import sys
import warnings

# PyTorch reads PYTORCH_ENABLE_MPS_FALLBACK when it is first imported, so the flag has to be set before anything imports
# torch. It lets operations that are not yet implemented for the 'mps' device run on the CPU instead of raising an
# error. A value that is already in the environment is respected, so users can opt out with
# PYTORCH_ENABLE_MPS_FALLBACK=0.
if 'torch' in sys.modules and 'PYTORCH_ENABLE_MPS_FALLBACK' not in os.environ:
    import torch
    if torch.backends.mps.is_available():
        warnings.warn("torch was imported before Winnie, so Winnie could not set PYTORCH_ENABLE_MPS_FALLBACK=1. "
                      "Operations that are unsupported on 'mps' will raise errors instead of falling back to the CPU. "
                      "Import winnie before torch or set the variable before starting Python.", stacklevel=2)
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

from .space import SpaceRDI, SpaceReduction, SpaceConvolution
from . import backend
from .backend import get_default_device, set_default_device
from . import rdi
from . import utils
from . import convolution
from . import plot