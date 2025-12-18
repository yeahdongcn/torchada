#!/usr/bin/env python3
"""
Example setup.py for building a C++/CUDA extension with torchada.

This example shows how to build CUDA extensions that work on both
CUDA and MUSA platforms using standard torch imports.

Key point: Just import torchada first, then use standard torch imports.
No code changes needed!

Usage:
    python examples/extension_setup.py build_ext --inplace
"""

import os
from setuptools import setup, find_packages

# Import torchada first to apply patches
import torchada  # noqa: F401

# Now use standard torch imports - they work on both CUDA and MUSA!
from torch.utils.cpp_extension import (
    CUDAExtension,
    CppExtension,
    BuildExtension,
    CUDA_HOME,
)
from torchada import detect_platform, Platform


def get_extensions():
    """Build the list of extensions."""
    extensions = []

    # Get current platform
    platform = detect_platform()
    print(f"Building for platform: {platform.value}")
    print(f"CUDA/MUSA home: {CUDA_HOME}")

    # Define source files
    # In a real project, these would be actual .cpp and .cu files
    sources = [
        # "src/my_extension.cpp",
        # "src/my_kernel.cu",
    ]

    # Skip if no source files (this is just an example)
    if not sources:
        print("No source files found, skipping extension build.")
        print("This is just an example showing the setup structure.")
        return extensions

    # Common compile flags
    cxx_flags = ["-O3", "-std=c++17"]

    # Platform-specific GPU flags
    if platform == Platform.MUSA:
        # MUSA compiler flags
        gpu_flags = ["-O3", "-std=c++17"]
        gpu_key = "mcc"  # MUSA compiler
    else:
        # CUDA compiler flags
        gpu_flags = [
            "-O3",
            "-std=c++17",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
        ]
        gpu_key = "nvcc"

    # Create the extension
    ext = CUDAExtension(
        name="my_extension",
        sources=sources,
        extra_compile_args={
            "cxx": cxx_flags,
            gpu_key: gpu_flags,
        },
        include_dirs=[
            # Add your include directories here
            # os.path.join(os.path.dirname(__file__), "include"),
        ],
    )
    extensions.append(ext)

    return extensions


# Example of how to set up the package
if __name__ == "__main__":
    print("=" * 60)
    print("torchada Extension Build Example")
    print("=" * 60)

    extensions = get_extensions()

    if extensions:
        setup(
            name="my_cuda_extension",
            version="0.1.0",
            ext_modules=extensions,
            cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
            python_requires=">=3.8",
        )
    else:
        print("\nTo build an actual extension, add source files to the sources list.")
        print("\nExample directory structure:")
        print("  my_project/")
        print("  ├── setup.py              # Use this as a template")
        print("  ├── src/")
        print("  │   ├── my_extension.cpp  # C++ bindings")
        print("  │   └── my_kernel.cu      # CUDA/MUSA kernels")
        print("  └── my_extension/")
        print("      └── __init__.py       # Python package")

