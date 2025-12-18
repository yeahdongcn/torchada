"""
Platform detection module for torchada.

Detects whether the current environment supports CUDA (NVIDIA) or MUSA (Moore Threads).
"""

import os
from enum import Enum
from functools import lru_cache
from typing import Optional


class Platform(Enum):
    """Supported GPU platforms."""
    CUDA = "cuda"
    MUSA = "musa"
    CPU = "cpu"


@lru_cache(maxsize=1)
def detect_platform() -> Platform:
    """
    Detect the current GPU platform.
    
    Priority:
    1. TORCHADA_PLATFORM environment variable (force specific platform)
    2. MUSA availability (Moore Threads GPU)
    3. CUDA availability (NVIDIA GPU)
    4. CPU fallback
    
    Returns:
        Platform: The detected or configured platform.
    """
    # Check for forced platform via environment variable
    forced_platform = os.environ.get("TORCHADA_PLATFORM", "").lower()
    if forced_platform == "cuda":
        return Platform.CUDA
    elif forced_platform == "musa":
        return Platform.MUSA
    elif forced_platform == "cpu":
        return Platform.CPU
    
    # Auto-detect platform
    # Check MUSA first (Moore Threads)
    if _is_musa_available():
        return Platform.MUSA
    
    # Check CUDA (NVIDIA)
    if _is_cuda_available():
        return Platform.CUDA
    
    # Fallback to CPU
    return Platform.CPU


def _is_musa_available() -> bool:
    """Check if MUSA (Moore Threads) is available."""
    try:
        import torch_musa
        import torch
        return torch.musa.is_available()
    except (ImportError, AttributeError):
        return False


def _is_cuda_available() -> bool:
    """Check if CUDA (NVIDIA) is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except (ImportError, AttributeError):
        return False


def is_musa_platform() -> bool:
    """Check if we're on MUSA platform."""
    return detect_platform() == Platform.MUSA


def is_cuda_platform() -> bool:
    """Check if we're on CUDA platform."""
    return detect_platform() == Platform.CUDA


def is_cpu_platform() -> bool:
    """Check if we're on CPU-only platform."""
    return detect_platform() == Platform.CPU


def get_device_name() -> str:
    """Get the device name string ('cuda', 'musa', or 'cpu')."""
    return detect_platform().value


def get_torch_device_module():
    """
    Get the appropriate torch device module (torch.cuda or torch.musa).
    
    Returns:
        The torch.cuda or torch.musa module.
    
    Raises:
        RuntimeError: If no GPU platform is available.
    """
    platform = detect_platform()
    
    if platform == Platform.MUSA:
        import torch_musa
        import torch
        return torch.musa
    elif platform == Platform.CUDA:
        import torch
        return torch.cuda
    else:
        raise RuntimeError("No GPU platform available. Running on CPU only.")

