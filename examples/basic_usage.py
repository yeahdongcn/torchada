#!/usr/bin/env python3
"""
Basic usage example for torchada.

This example shows how torchada makes your existing torch.cuda code
work transparently on any supported GPU platform (CUDA or MUSA).

Usage:
    Just import torchada at the top of your script, then use
    torch.cuda.* APIs as you normally would.

Note:
    torch.cuda.is_available() returns False on MUSA platform (by design).
    Use torchada.is_musa_platform() or torch.musa.is_available() for MUSA.
"""

import torch

import torchada  # noqa: F401 - Import first to apply patches


def is_gpu_available():
    """Check if any GPU (CUDA or MUSA) is available."""
    return torchada.is_musa_platform() or torch.cuda.is_available()


def main():
    # Check for GPU availability (works on both CUDA and MUSA)
    if is_gpu_available():
        print("GPU is available!")
        print(f"  Platform: {torchada.get_platform().name}")
        print(f"  Device count: {torch.cuda.device_count()}")
        print(f"  Current device: {torch.cuda.current_device()}")
        print(f"  Device name: {torch.cuda.get_device_name()}")

        # Create tensor on GPU using standard .cuda() method
        # On MUSA platform, this creates a tensor on MUSA device
        x = torch.randn(1000, 1000).cuda()
        print(f"\nCreated tensor on GPU: {x.device}")

        # Alternative: use .to("cuda") - also works on MUSA
        y = torch.randn(1000, 1000).to("cuda")
        print(f"Created another tensor: {y.device}")

        # Matrix multiplication on GPU
        try:
            z = torch.matmul(x, y)
            print(f"Result tensor: {z.device}, shape: {z.shape}")
        except RuntimeError as e:
            print(
                f"Matrix multiplication skipped (driver/hardware issue): {type(e).__name__}"
            )
            z = None

        # Memory info using standard torch.cuda API
        print(f"\nMemory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"Memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

        # Synchronize
        torch.cuda.synchronize()
        print("\nGPU synchronized successfully!")

        # Clean up
        del x, y
        if z is not None:
            del z
        torch.cuda.empty_cache()
        print("Cache cleared.")
    else:
        print("No GPU available, running on CPU.")
        x = torch.randn(1000, 1000)
        print(f"Created tensor on CPU: {x.device}")


if __name__ == "__main__":
    main()
