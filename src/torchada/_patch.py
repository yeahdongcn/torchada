"""
Automatic patching module for torchada.

This module patches PyTorch to automatically translate 'cuda' device strings
to 'musa' when running on Moore Threads hardware.

Usage:
    import torchada  # This applies all patches automatically
    import torch

    # Then use torch.cuda as normal - it will work on MUSA
    torch.cuda.is_available()
    x = torch.randn(3, 3).cuda()
    from torch.cuda.amp import autocast, GradScaler

    # Distributed training with NCCL also works transparently
    import torch.distributed as dist
    dist.init_process_group(backend="nccl")  # Uses MCCL on MUSA

    # CUDA Graphs work transparently
    g = torch.cuda.CUDAGraph()  # Uses MUSAGraph on MUSA
"""

import functools
import sys
from types import ModuleType
from typing import Any, Callable, Optional, Union

import torch

from ._platform import is_musa_platform, get_device_name


_patched = False
_original_init_process_group = None


def _translate_device(device: Any) -> Any:
    """
    Translate 'cuda' device references to 'musa' on MUSA platform.

    Args:
        device: Device specification (string, torch.device, int, or None)

    Returns:
        Translated device specification
    """
    if not is_musa_platform():
        return device

    if device is None:
        return device

    if isinstance(device, str):
        # Handle 'cuda', 'cuda:0', 'cuda:1', etc.
        if device == "cuda" or device.startswith("cuda:"):
            return device.replace("cuda", "musa")
        return device

    if isinstance(device, torch.device):
        if device.type == "cuda":
            return torch.device("musa", device.index)
        return device

    # For integer device IDs, keep as-is (context determines device type)
    return device


def _wrap_to_method(original_to: Callable) -> Callable:
    """Wrap tensor.to() to translate device strings."""
    @functools.wraps(original_to)
    def wrapped_to(self, *args, **kwargs):
        # Translate device in positional args
        if args and len(args) >= 1:
            first_arg = args[0]
            # Check if first arg looks like a device
            if isinstance(first_arg, (str, torch.device)):
                args = (_translate_device(first_arg),) + args[1:]
            elif isinstance(first_arg, torch.dtype):
                # .to(dtype) case, check for device in kwargs or second arg
                if len(args) >= 2:
                    args = (first_arg, _translate_device(args[1])) + args[2:]

        # Translate device in keyword args
        if "device" in kwargs:
            kwargs["device"] = _translate_device(kwargs["device"])

        return original_to(self, *args, **kwargs)

    return wrapped_to


def _wrap_tensor_cuda(original_cuda: Callable) -> Callable:
    """Wrap tensor.cuda() to use musa on MUSA platform."""
    @functools.wraps(original_cuda)
    def wrapped_cuda(self, device=None, non_blocking=False):
        if is_musa_platform():
            # Use .musa() instead
            if hasattr(self, 'musa'):
                return self.musa(device=device, non_blocking=non_blocking)
            else:
                # Fallback to .to()
                target_device = f"musa:{device}" if device is not None else "musa"
                return self.to(target_device, non_blocking=non_blocking)
        return original_cuda(self, device=device, non_blocking=non_blocking)

    return wrapped_cuda


def _wrap_module_cuda(original_cuda: Callable) -> Callable:
    """Wrap nn.Module.cuda() to use musa on MUSA platform."""
    @functools.wraps(original_cuda)
    def wrapped_cuda(self, device=None):
        if is_musa_platform():
            if hasattr(self, 'musa'):
                return self.musa(device=device)
            else:
                target_device = f"musa:{device}" if device is not None else "musa"
                return self.to(target_device)
        return original_cuda(self, device=device)

    return wrapped_cuda


def _wrap_torch_device(original_device: type) -> type:
    """Wrap torch.device constructor to translate cuda to musa."""
    class WrappedDevice(original_device):
        def __new__(cls, device, index=None):
            if isinstance(device, str):
                device = _translate_device(device)
            return super().__new__(cls, device, index)

    return WrappedDevice


def _wrap_factory_function(original_fn: Callable) -> Callable:
    """Wrap tensor factory functions (empty, zeros, ones, etc.) to translate device."""
    @functools.wraps(original_fn)
    def wrapped_fn(*args, **kwargs):
        if "device" in kwargs:
            kwargs["device"] = _translate_device(kwargs["device"])
        return original_fn(*args, **kwargs)
    return wrapped_fn


# List of torch factory functions that accept a device argument
_FACTORY_FUNCTIONS = [
    'empty', 'zeros', 'ones', 'full', 'rand', 'randn', 'randint',
    'arange', 'linspace', 'logspace', 'eye', 'tensor',
    'as_tensor', 'from_numpy', 'empty_like', 'zeros_like', 'ones_like',
    'full_like', 'rand_like', 'randn_like', 'randint_like',
    'empty_strided', 'sparse_coo_tensor', 'sparse_csr_tensor',
]


def _patch_torch_cuda_module():
    """
    Patch torch.cuda to redirect to torch.musa on MUSA platform.

    This allows developers to use torch.cuda.* APIs transparently.
    """
    try:
        import torch_musa
    except ImportError:
        return

    # torch_musa registers itself as torch.musa when imported
    # Now patch torch.cuda to point to torch.musa (which is torch_musa)
    if hasattr(torch, 'musa'):
        # Replace torch.cuda with torch.musa in sys.modules
        # This makes 'from torch.cuda import ...' work
        sys.modules['torch.cuda'] = torch.musa

        # Also patch torch.cuda attribute directly
        torch.cuda = torch.musa

        # Patch torch.cuda.amp
        if hasattr(torch.musa, 'amp'):
            sys.modules['torch.cuda.amp'] = torch.musa.amp

        # Patch torch.cuda.graphs - MUSAGraph should be accessible as CUDAGraph
        if hasattr(torch.musa, 'graphs'):
            sys.modules['torch.cuda.graphs'] = torch.musa.graphs

        # Add CUDAGraph alias pointing to MUSAGraph
        if hasattr(torch.musa, 'MUSAGraph') and not hasattr(torch.musa, 'CUDAGraph'):
            torch.musa.CUDAGraph = torch.musa.MUSAGraph

        # Patch torch.cuda.nccl -> torch.musa.mccl
        if hasattr(torch.musa, 'mccl'):
            sys.modules['torch.cuda.nccl'] = torch.musa.mccl

        # Patch torch.cuda.profiler
        if hasattr(torch.musa, 'profiler'):
            sys.modules['torch.cuda.profiler'] = torch.musa.profiler


def _patch_distributed_backend():
    """
    Patch torch.distributed to automatically use MCCL when NCCL is requested.

    This allows code using 'nccl' backend to work transparently on MUSA.
    """
    global _original_init_process_group

    try:
        import torch.distributed as dist
    except ImportError:
        return

    if _original_init_process_group is not None:
        # Already patched
        return

    _original_init_process_group = dist.init_process_group

    @functools.wraps(_original_init_process_group)
    def patched_init_process_group(
        backend: Optional[str] = None,
        init_method: Optional[str] = None,
        timeout=None,
        world_size: int = -1,
        rank: int = -1,
        store=None,
        group_name: str = '',
        pg_options=None,
        device_id=None,
    ):
        # Translate 'nccl' to 'mccl' on MUSA platform
        if is_musa_platform() and backend is not None:
            if backend.lower() == 'nccl':
                backend = 'mccl'

        # Translate device_id if it's a cuda device
        if device_id is not None:
            device_id = _translate_device(device_id)

        # Build kwargs for the original function
        kwargs = {
            'backend': backend,
            'init_method': init_method,
            'world_size': world_size,
            'rank': rank,
            'store': store,
            'group_name': group_name,
            'pg_options': pg_options,
            'device_id': device_id,
        }
        if timeout is not None:
            kwargs['timeout'] = timeout

        return _original_init_process_group(**kwargs)

    dist.init_process_group = patched_init_process_group


def _patch_nccl_module():
    """
    Create torch.cuda.nccl module that redirects to MCCL.

    This allows code importing torch.cuda.nccl to work on MUSA.
    """
    try:
        import torch_musa
    except ImportError:
        return

    if hasattr(torch_musa, 'mccl'):
        # Create a wrapper module for nccl -> mccl
        sys.modules['torch.cuda.nccl'] = torch_musa.mccl


def _patch_is_cuda_available():
    """
    Patch torch.cuda.is_available to return MUSA availability.

    This is a safety net in case the module swap doesn't work perfectly.
    """
    # This is now handled by module redirection, but we keep it for reference
    pass


def _patch_autocast():
    """
    Ensure torch.amp.autocast works with 'cuda' device_type on MUSA.
    """
    try:
        import torch_musa
    except ImportError:
        return

    if not hasattr(torch, 'amp') or not hasattr(torch.amp, 'autocast'):
        return

    original_autocast = torch.amp.autocast

    class PatchedAutocast(original_autocast):
        def __init__(self, device_type, *args, **kwargs):
            # Translate 'cuda' to 'musa'
            if device_type == 'cuda':
                device_type = 'musa'
            super().__init__(device_type, *args, **kwargs)

    torch.amp.autocast = PatchedAutocast


def apply_patches():
    """
    Apply all necessary patches for CUDA to MUSA translation.

    After calling this, developers can use torch.cuda.* APIs normally
    and they will be transparently redirected to torch.musa on MUSA platform.

    This includes:
    - torch.cuda.* API -> torch.musa.*
    - torch.Tensor.cuda() -> torch.Tensor.musa()
    - torch.nn.Module.cuda() -> torch.nn.Module.musa()
    - Device string translation ("cuda" -> "musa")
    - torch.distributed with 'nccl' backend -> 'mccl'
    - torch.cuda.CUDAGraph -> torch.musa.MUSAGraph
    - torch.cuda.nccl -> torch.musa.mccl
    - torch.amp.autocast(device_type='cuda') -> 'musa'

    This function should be called once at import time.
    """
    global _patched

    if _patched:
        return

    if not is_musa_platform():
        _patched = True
        return

    # Import torch_musa to ensure it's initialized
    try:
        import torch_musa
    except ImportError:
        _patched = True
        return

    # Patch torch.cuda module to redirect to torch.musa
    _patch_torch_cuda_module()

    # Patch torch.distributed to use MCCL when NCCL is requested
    _patch_distributed_backend()

    # Patch torch.cuda.nccl module
    _patch_nccl_module()

    # Patch torch.amp.autocast for device_type translation
    _patch_autocast()

    # Patch torch.Tensor.to()
    if hasattr(torch.Tensor, 'to'):
        torch.Tensor.to = _wrap_to_method(torch.Tensor.to)

    # Patch torch.Tensor.cuda()
    if hasattr(torch.Tensor, 'cuda'):
        torch.Tensor.cuda = _wrap_tensor_cuda(torch.Tensor.cuda)

    # Patch torch.nn.Module.cuda()
    if hasattr(torch.nn.Module, 'cuda'):
        torch.nn.Module.cuda = _wrap_module_cuda(torch.nn.Module.cuda)

    # Patch tensor factory functions
    for fn_name in _FACTORY_FUNCTIONS:
        if hasattr(torch, fn_name):
            original_fn = getattr(torch, fn_name)
            setattr(torch, fn_name, _wrap_factory_function(original_fn))

    _patched = True


def is_patched() -> bool:
    """Check if patches have been applied."""
    return _patched


# Additional exports for advanced usage
def get_original_init_process_group():
    """Get the original torch.distributed.init_process_group function."""
    return _original_init_process_group

