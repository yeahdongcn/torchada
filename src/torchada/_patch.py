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


_original_torch_device = None


class _DeviceFactory:
    """
    A callable class that wraps torch.device to translate 'cuda' to 'musa'.

    This class mimics torch.device behavior while translating device strings.
    It maintains isinstance() compatibility by returning actual torch.device objects.
    """

    def __init__(self, original_device_class):
        self._original = original_device_class
        # Copy class attributes for compatibility
        self.__doc__ = original_device_class.__doc__
        self.__name__ = 'device'
        self.__qualname__ = 'device'
        self.__module__ = 'torch'

    def __call__(self, device=None, index=None):
        # Handle the case where device is already a torch.device
        if isinstance(device, self._original):
            if device.type == "cuda":
                device = "musa"
                index = device.index if index is None else index
            else:
                return device

        # Handle string device
        if isinstance(device, str):
            device = _translate_device(device)

        # Create the actual device
        if index is not None:
            return self._original(device, index)
        elif device is not None:
            return self._original(device)
        else:
            return self._original()

    def __instancecheck__(cls, instance):
        """Make isinstance(x, torch.device) work correctly."""
        return isinstance(instance, cls._original)

    # Make this class work with isinstance() by implementing __class__
    @property
    def __class__(self):
        return type(self._original)


class _DeviceFactoryMeta(type):
    """Metaclass to make isinstance(x, torch.device) work with our factory."""

    def __instancecheck__(cls, instance):
        if _original_torch_device is not None:
            return isinstance(instance, _original_torch_device)
        return False

    def __subclasscheck__(cls, subclass):
        if _original_torch_device is not None:
            return issubclass(subclass, _original_torch_device)
        return False


class DeviceFactoryWrapper(metaclass=_DeviceFactoryMeta):
    """
    A wrapper class that acts as torch.device but translates cuda to musa.

    Uses a metaclass to properly handle isinstance() checks.
    """
    _original = None

    def __new__(cls, device=None, index=None):
        original = cls._original
        if original is None:
            raise RuntimeError("DeviceFactoryWrapper not initialized")

        # Handle the case where device is already a torch.device
        if isinstance(device, original):
            if device.type == "cuda":
                device = "musa"
                index = device.index if index is None else index
            else:
                return device

        # Handle string device
        if isinstance(device, str):
            device = _translate_device(device)

        # Create the actual device
        if index is not None:
            return original(device, index)
        elif device is not None:
            return original(device)
        else:
            return original()


def _patch_torch_device():
    """
    Patch torch.device to translate 'cuda' to 'musa' on MUSA platform.

    This ensures that torch.device("cuda:0") creates a musa device when on MUSA.
    """
    global _original_torch_device

    if _original_torch_device is not None:
        return  # Already patched

    _original_torch_device = torch.device
    DeviceFactoryWrapper._original = _original_torch_device

    # Replace torch.device with our wrapper
    torch.device = DeviceFactoryWrapper


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

        # Patch torch.cuda.nvtx - use our stub since MUSA doesn't have nvtx
        try:
            from .cuda import nvtx as nvtx_stub
            sys.modules['torch.cuda.nvtx'] = nvtx_stub
            torch.musa.nvtx = nvtx_stub
        except ImportError:
            pass


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


def _patch_torch_version():
    """
    Patch torch.version.cuda to return MUSA version on MUSA platform.

    This allows code that checks torch.version.cuda to work on MUSA.
    """
    if hasattr(torch.version, 'musa') and torch.version.musa is not None:
        # Set torch.version.cuda to the MUSA version
        torch.version.cuda = str(torch.version.musa)


def _patch_tensor_is_cuda():
    """
    Patch torch.Tensor.is_cuda property to return True for MUSA tensors.

    This allows code that checks tensor.is_cuda to work on MUSA.
    We patch the is_cuda property to also return True for MUSA tensors.
    """
    # Store the original is_cuda property (it's a getset_descriptor)
    original_is_cuda = torch.Tensor.is_cuda

    @property
    def patched_is_cuda(self):
        """Return True if tensor is on CUDA or MUSA device."""
        # Check original is_cuda first
        try:
            if original_is_cuda.__get__(self):
                return True
        except Exception:
            pass
        # Also return True for MUSA tensors
        return getattr(self, 'is_musa', False)

    # Replace is_cuda with our patched version
    torch.Tensor.is_cuda = patched_is_cuda


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


def _patch_cpp_extension():
    """
    Patch torch.utils.cpp_extension to use torchada's MUSA-compatible versions.

    This allows developers to use standard imports like:
        from torch.utils.cpp_extension import CUDAExtension, BuildExtension

    And have them work transparently on MUSA platform.
    """
    try:
        # Import our cpp_extension module which has the MUSA-compatible implementations
        from .utils import cpp_extension as torchada_cpp_ext
    except ImportError:
        return

    # Get the original torch.utils.cpp_extension module
    import torch.utils.cpp_extension as torch_cpp_ext

    # Patch the key classes and functions
    torch_cpp_ext.CUDAExtension = torchada_cpp_ext.CUDAExtension
    torch_cpp_ext.BuildExtension = torchada_cpp_ext.BuildExtension
    torch_cpp_ext.CUDA_HOME = torchada_cpp_ext.CUDA_HOME

    # Also update sys.modules entry
    sys.modules['torch.utils.cpp_extension'] = torch_cpp_ext


def apply_patches():
    """
    Apply all necessary patches for CUDA to MUSA translation.

    After calling this, developers can use torch.cuda.* APIs normally
    and they will be transparently redirected to torch.musa on MUSA platform.

    This includes:
    - torch.device("cuda") -> torch.device("musa")
    - torch.version.cuda -> MUSA version string
    - torch.cuda.* API -> torch.musa.*
    - torch.cuda.nvtx -> no-op stub
    - torch.Tensor.cuda() -> torch.Tensor.musa()
    - torch.Tensor.is_cuda_compat property (True for CUDA or MUSA tensors)
    - torch.nn.Module.cuda() -> torch.nn.Module.musa()
    - Device string translation ("cuda" -> "musa")
    - torch.distributed with 'nccl' backend -> 'mccl'
    - torch.cuda.CUDAGraph -> torch.musa.MUSAGraph
    - torch.cuda.nccl -> torch.musa.mccl
    - torch.amp.autocast(device_type='cuda') -> 'musa'
    - torch.utils.cpp_extension (CUDAExtension, BuildExtension) -> MUSA versions

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

    # Patch torch.device to translate cuda to musa
    _patch_torch_device()

    # Patch torch.version.cuda to return MUSA version
    _patch_torch_version()

    # Patch torch.Tensor to add is_cuda_compat property
    _patch_tensor_is_cuda()

    # Patch torch.cuda module to redirect to torch.musa
    _patch_torch_cuda_module()

    # Patch torch.distributed to use MCCL when NCCL is requested
    _patch_distributed_backend()

    # Patch torch.cuda.nccl module
    _patch_nccl_module()

    # Patch torch.amp.autocast for device_type translation
    _patch_autocast()

    # Patch torch.utils.cpp_extension for MUSA compatibility
    _patch_cpp_extension()

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

