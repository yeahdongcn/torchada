"""
torchada - Adapter package for torch_musa to act exactly like PyTorch CUDA.

torchada provides a unified interface that works transparently on both
NVIDIA GPUs (CUDA) and Moore Threads GPUs (MUSA).

Usage:
    Just import torchada at the top of your script, then use standard
    torch.cuda.* and torch.utils.cpp_extension APIs as you normally would.
    torchada patches PyTorch to transparently redirect to MUSA on
    Moore Threads hardware.

    # Add this at the top of your script:
    import torchada  # noqa: F401

    # Then use standard torch APIs - they work on MUSA too!
    import torch
    torch.cuda.is_available()
    x = torch.randn(3, 3).cuda()
    from torch.cuda.amp import autocast, GradScaler
    from torch.utils.cpp_extension import CUDAExtension, BuildExtension, CUDA_HOME
"""

__version__ = "0.1.0"

from . import cuda
from . import utils
from ._platform import (
    Platform,
    detect_platform,
    is_musa_platform,
    is_cuda_platform,
    is_cpu_platform,
    get_device_name,
    get_torch_device_module,
)
from ._patch import apply_patches, is_patched, get_original_init_process_group
from .utils.cpp_extension import CUDA_HOME


# Automatically apply patches on import
apply_patches()


def get_version() -> str:
    """Return the version of torchada."""
    return __version__


def get_platform() -> Platform:
    """Return the detected platform."""
    return detect_platform()


def get_backend():
    """
    Get the underlying torch device module (torch.cuda or torch.musa).

    Returns:
        The torch.cuda or torch.musa module.
    """
    return get_torch_device_module()


__all__ = [
    # Version
    "__version__",
    "get_version",
    # Modules
    "cuda",
    "utils",
    # Platform detection
    "Platform",
    "detect_platform",
    "is_musa_platform",
    "is_cuda_platform",
    "is_cpu_platform",
    "get_device_name",
    "get_platform",
    "get_backend",
    # Patching
    "apply_patches",
    "is_patched",
    "get_original_init_process_group",
    # C++ Extension building
    "CUDA_HOME",
]

