"""
torchada.utils.cpp_extension - C++/CUDA extension utilities.

This module provides CUDAExtension, BuildExtension, and related utilities
that work on both CUDA and MUSA platforms.

Note: After importing torchada, you can use standard torch.utils.cpp_extension
imports - they are automatically patched to use these implementations on MUSA.

Usage (preferred):
    import torchada  # Apply patches first
    from torch.utils.cpp_extension import CUDAExtension, BuildExtension, CUDA_HOME

    ext_modules = [
        CUDAExtension(
            name="my_extension",
            sources=["my_extension.cpp", "my_extension_kernel.cu"],
        )
    ]

    setup(
        name="my_package",
        ext_modules=ext_modules,
        cmdclass={"build_ext": BuildExtension},
    )
"""

import os
import re
from typing import Any, Dict, List, Optional

from .._mapping import _MAPPING_RULE, EXT_REPLACED_MAPPING
from .._platform import Platform, detect_platform, is_musa_platform

# Flag to track if torch_musa patches have been applied
_musa_patches_applied = False


def _get_cuda_home() -> Optional[str]:
    """
    Get the CUDA or MUSA home directory.

    On MUSA platform, this returns MUSA_HOME but is still called CUDA_HOME
    so developers don't need to change their code.
    """
    platform = detect_platform()

    if platform == Platform.MUSA:
        # Check MUSA_HOME first, then common paths
        musa_home = os.environ.get("MUSA_HOME")
        if musa_home:
            return musa_home

        # Common MUSA installation paths
        common_paths = [
            "/usr/local/musa",
            "/opt/musa",
        ]
        for path in common_paths:
            if os.path.exists(path):
                return path
        return None

    elif platform == Platform.CUDA:
        # Use torch's CUDA_HOME
        try:
            from torch.utils.cpp_extension import CUDA_HOME as TORCH_CUDA_HOME

            return TORCH_CUDA_HOME
        except ImportError:
            cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
            if cuda_home:
                return cuda_home
            if os.path.exists("/usr/local/cuda"):
                return "/usr/local/cuda"
            return None

    return None


def _is_cuda_file(path: str) -> bool:
    """Check if a file is a CUDA source file."""
    ext = os.path.splitext(path)[1]
    return ext in [".cu", ".cuh"]


def _is_musa_file(path: str) -> bool:
    """
    Check if a file is a MUSA source file.
    Also recognizes .cu/.cuh files as MUSA files for compatibility.

    This function is used to patch torch_musa so it recognizes .cu files.
    """
    ext = os.path.splitext(path)[1]
    return ext in [".cu", ".cuh", ".mu", ".muh"]


def _apply_musa_patches():
    """
    Apply patches to torch_musa modules for CUDA compatibility.

    This function patches:
    1. musa_ext._is_musa_file - to recognize .cu/.cuh files
    2. musa_sp.EXT_REPLACED_MAPPING - to convert .cu/.cuh to .mu/.muh
    3. musa_sp._MAPPING_RULE - to apply CUDA->MUSA symbol mapping

    These patches are required to compile .cu files on MUSA platform.
    """
    global _musa_patches_applied

    if _musa_patches_applied:
        return

    if not is_musa_platform():
        return

    try:
        import torch_musa.utils.musa_extension as musa_ext
        import torch_musa.utils.simple_porting as musa_sp

        # Patch _is_musa_file to recognize .cu/.cuh files
        musa_ext._is_musa_file = _is_musa_file

        # Patch EXT_REPLACED_MAPPING to convert .cu/.cuh to .mu/.muh
        musa_sp.EXT_REPLACED_MAPPING = EXT_REPLACED_MAPPING

        # Patch _MAPPING_RULE with our comprehensive CUDA->MUSA mappings
        # This is the critical patch that enables source code porting
        musa_sp._MAPPING_RULE = _MAPPING_RULE

        _musa_patches_applied = True

    except ImportError:
        # torch_musa not available, patches not needed
        pass


# Apply MUSA patches at module import time
_apply_musa_patches()

# Export CUDA_HOME - always use this name, even on MUSA platform
# This way developers don't need to change their code
CUDA_HOME = _get_cuda_home()


def _port_cuda_source(
    source_code: str, mapping_rules: Optional[Dict[str, str]] = None
) -> str:
    """
    Port CUDA source code to MUSA by applying mapping rules.

    Args:
        source_code: The CUDA source code to port
        mapping_rules: Optional custom mapping rules (defaults to _MAPPING_RULE)

    Returns:
        The ported MUSA source code
    """
    if mapping_rules is None:
        mapping_rules = _MAPPING_RULE

    result = source_code

    # Sort rules by length (longest first) to avoid partial replacements
    sorted_rules = sorted(mapping_rules.items(), key=lambda x: len(x[0]), reverse=True)

    for cuda_symbol, musa_symbol in sorted_rules:
        if cuda_symbol != musa_symbol:
            result = result.replace(cuda_symbol, musa_symbol)

    return result


def include_paths(cuda: bool = True) -> List[str]:
    """
    Get include paths for compiling extensions.

    Args:
        cuda: Whether to include CUDA/MUSA paths

    Returns:
        List of include paths
    """
    platform = detect_platform()

    if platform == Platform.MUSA:
        try:
            import torch_musa.utils.musa_extension as musa_ext

            if hasattr(musa_ext, "include_paths"):
                return musa_ext.include_paths(cuda=cuda)
        except ImportError:
            pass

        # Fallback: construct paths manually
        paths = []
        musa_home = _get_cuda_home()
        if musa_home:
            paths.append(os.path.join(musa_home, "include"))
        return paths

    else:
        from torch.utils.cpp_extension import include_paths as torch_include_paths

        return torch_include_paths(cuda=cuda)


def library_paths(cuda: bool = True) -> List[str]:
    """
    Get library paths for compiling extensions.

    Args:
        cuda: Whether to include CUDA/MUSA library paths

    Returns:
        List of library paths
    """
    platform = detect_platform()

    if platform == Platform.MUSA:
        paths = []
        musa_home = _get_cuda_home()
        if musa_home:
            paths.append(os.path.join(musa_home, "lib"))
            paths.append(os.path.join(musa_home, "lib64"))
        return [p for p in paths if os.path.exists(p)]

    else:
        from torch.utils.cpp_extension import library_paths as torch_library_paths

        return torch_library_paths(cuda=cuda)


class CUDAExtension:
    """
    A wrapper that creates either a torch CUDAExtension or MUSA MUSAExtension.

    This class provides a unified interface for building CUDA extensions
    that works transparently on both CUDA and MUSA platforms.
    """

    def __new__(cls, name: str, sources: List[str], *args, **kwargs):
        """
        Create a new extension module.

        Args:
            name: The name of the extension
            sources: List of source files
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        platform = detect_platform()

        if platform == Platform.MUSA:
            return _create_musa_extension(name, sources, *args, **kwargs)
        else:
            return _create_cuda_extension(name, sources, *args, **kwargs)


class CppExtension:
    """
    A wrapper for creating C++ extensions (no CUDA/MUSA).
    """

    def __new__(cls, name: str, sources: List[str], *args, **kwargs):
        """Create a C++ extension module."""
        from torch.utils.cpp_extension import CppExtension as TorchCppExtension

        return TorchCppExtension(name, sources, *args, **kwargs)


def _create_cuda_extension(name: str, sources: List[str], *args, **kwargs):
    """Create a CUDA extension using torch's CUDAExtension."""
    from torch.utils.cpp_extension import CUDAExtension as TorchCUDAExtension

    return TorchCUDAExtension(name, sources, *args, **kwargs)


def _create_musa_extension(name: str, sources: List[str], *args, **kwargs):
    """Create a MUSA extension using torch_musa's MUSAExtension.

    The patches applied by _apply_musa_patches() make MUSAExtension accept
    .cu/.cuh files directly by:
    1. Patching musa_ext._is_musa_file to recognize .cu/.cuh as valid MUSA files
    2. Patching musa_sp.EXT_REPLACED_MAPPING to convert .cu/.cuh to .mu/.muh
    3. Patching musa_sp._MAPPING_RULE to convert CUDA symbols to MUSA in source code
    """
    # Ensure patches are applied
    _apply_musa_patches()

    try:
        import torch_musa.utils.musa_extension as musa_ext

        # Simply pass sources to MUSAExtension - patches make it accept .cu files
        return musa_ext.MUSAExtension(name, sources, *args, **kwargs)
    except ImportError:
        # Fallback to torch's CUDAExtension if torch_musa is not available
        from torch.utils.cpp_extension import CUDAExtension as TorchCUDAExtension

        return TorchCUDAExtension(name, sources, *args, **kwargs)


def _get_build_extension_class():
    """
    Get the BuildExtension class for the current platform.

    On MUSA platform, returns a custom class that:
    1. Uses SimplePorting to convert CUDA sources to MUSA in run()
    2. Registers .cu/.cuh as valid source extensions in build_extensions()
    """
    platform = detect_platform()

    if platform == Platform.MUSA:
        # Ensure patches are applied
        _apply_musa_patches()
        try:
            import torch_musa.utils.musa_extension as musa_ext
            import torch_musa.utils.simple_porting as musa_sp

            class _MUSABuildExtension(musa_ext.BuildExtension):
                """
                Custom BuildExtension that handles CUDA->MUSA source porting.

                Following the sglang pattern:
                - run(): Uses SimplePorting to convert source directories
                - build_extensions(): Registers .cu/.cuh as valid extensions
                """

                # Track directories that have been ported
                _ported_dirs = set()

                def build_extensions(self):
                    # Register .cu, .cuh as valid source extensions
                    self.compiler.src_extensions += [".cu", ".cuh"]
                    super().build_extensions()

                def run(self):
                    # Port all CUDA source directories before building
                    for ext in self.extensions:
                        new_sources = []
                        dirs_to_port = set()

                        for source in ext.sources:
                            source_path = os.path.abspath(source)
                            source_dir = os.path.dirname(source_path)
                            source_file = os.path.basename(source_path)
                            base_name, ext_name = os.path.splitext(source_file)
                            ext_name = ext_name.lower()

                            if ext_name in [".cu", ".cuh"]:
                                # Track this directory for porting
                                dirs_to_port.add(source_dir)
                                # Update source path to point to _musa directory
                                # SimplePorting converts .cu -> .mu, .cuh -> .muh
                                musa_dir = source_dir + "_musa"
                                new_ext = EXT_REPLACED_MAPPING.get(
                                    ext_name[1:], ext_name[1:]
                                )
                                new_source = os.path.join(
                                    musa_dir, base_name + "." + new_ext
                                )
                                new_sources.append(new_source)
                            else:
                                new_sources.append(source)

                        # Port each unique directory using our custom mapping rules
                        for cuda_dir in dirs_to_port:
                            if cuda_dir not in self._ported_dirs:
                                musa_sp.SimplePorting(
                                    cuda_dir_path=cuda_dir, mapping_rule=_MAPPING_RULE
                                ).run()
                                self._ported_dirs.add(cuda_dir)

                        # Update extension sources to point to ported files
                        ext.sources = new_sources

                    super().run()

            return _MUSABuildExtension
        except ImportError:
            pass

    # Fallback to torch's BuildExtension
    from torch.utils.cpp_extension import BuildExtension as TorchBuildExtension

    return TorchBuildExtension


# Get the actual BuildExtension class at module load time
# This ensures BuildExtension is a proper class that inherits from Command
BuildExtension = _get_build_extension_class()


def load(
    name: str,
    sources: List[str],
    extra_cflags: Optional[List[str]] = None,
    extra_cuda_cflags: Optional[List[str]] = None,
    extra_ldflags: Optional[List[str]] = None,
    extra_include_paths: Optional[List[str]] = None,
    build_directory: Optional[str] = None,
    verbose: bool = False,
    with_cuda: Optional[bool] = None,
    is_python_module: bool = True,
    is_standalone: bool = False,
    keep_intermediates: bool = True,
):
    """
    Load a PyTorch C++/CUDA extension at runtime (JIT compilation).

    This function works on both CUDA and MUSA platforms.

    Args:
        name: The name of the extension
        sources: List of source files
        extra_cflags: Extra C++ compiler flags
        extra_cuda_cflags: Extra CUDA/MUSA compiler flags
        extra_ldflags: Extra linker flags
        extra_include_paths: Extra include paths
        build_directory: Directory to build in
        verbose: Whether to print build output
        with_cuda: Whether to include CUDA/MUSA support
        is_python_module: Whether this is a Python module
        is_standalone: Whether this is a standalone executable
        keep_intermediates: Whether to keep intermediate files

    Returns:
        The loaded extension module
    """
    platform = detect_platform()

    if platform == Platform.MUSA:
        # Ensure patches are applied
        _apply_musa_patches()

        try:
            import torch_musa.utils.musa_extension as musa_ext

            # Use MUSA's load function if available
            if hasattr(musa_ext, "load"):
                return musa_ext.load(
                    name=name,
                    sources=sources,
                    extra_cflags=extra_cflags,
                    extra_cuda_cflags=extra_cuda_cflags,
                    extra_ldflags=extra_ldflags,
                    extra_include_paths=extra_include_paths,
                    build_directory=build_directory,
                    verbose=verbose,
                    with_cuda=with_cuda,
                    is_python_module=is_python_module,
                    is_standalone=is_standalone,
                    keep_intermediates=keep_intermediates,
                )
        except ImportError:
            pass

    # Fallback to torch's load
    from torch.utils.cpp_extension import load as torch_load

    return torch_load(
        name=name,
        sources=sources,
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        extra_ldflags=extra_ldflags,
        extra_include_paths=extra_include_paths,
        build_directory=build_directory,
        verbose=verbose,
        with_cuda=with_cuda,
        is_python_module=is_python_module,
        is_standalone=is_standalone,
        keep_intermediates=keep_intermediates,
    )


def load_inline(
    name: str,
    cpp_sources: List[str],
    cuda_sources: Optional[List[str]] = None,
    functions: Optional[List[str]] = None,
    extra_cflags: Optional[List[str]] = None,
    extra_cuda_cflags: Optional[List[str]] = None,
    extra_ldflags: Optional[List[str]] = None,
    extra_include_paths: Optional[List[str]] = None,
    build_directory: Optional[str] = None,
    verbose: bool = False,
    with_cuda: Optional[bool] = None,
    is_python_module: bool = True,
    with_pytorch_error_handling: bool = True,
    keep_intermediates: bool = True,
):
    """
    Load a PyTorch C++/CUDA extension from inline source code.

    This function works on both CUDA and MUSA platforms.
    """
    platform = detect_platform()

    # On MUSA platform, apply patches and port CUDA sources to MUSA
    if platform == Platform.MUSA:
        _apply_musa_patches()
        if cuda_sources:
            cuda_sources = [_port_cuda_source(src) for src in cuda_sources]

    from torch.utils.cpp_extension import load_inline as torch_load_inline

    return torch_load_inline(
        name=name,
        cpp_sources=cpp_sources,
        cuda_sources=cuda_sources,
        functions=functions,
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        extra_ldflags=extra_ldflags,
        extra_include_paths=extra_include_paths,
        build_directory=build_directory,
        verbose=verbose,
        with_cuda=with_cuda,
        is_python_module=is_python_module,
        with_pytorch_error_handling=with_pytorch_error_handling,
        keep_intermediates=keep_intermediates,
    )


# Export all public symbols
# Note: We only export CUDA_HOME, not MUSA_HOME. On MUSA platform, CUDA_HOME
# points to the MUSA installation so developers don't need to change their code.
__all__ = [
    "CUDA_HOME",
    "CUDAExtension",
    "CppExtension",
    "BuildExtension",
    "include_paths",
    "library_paths",
    "load",
    "load_inline",
]
