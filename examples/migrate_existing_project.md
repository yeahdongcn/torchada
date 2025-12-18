# Migrating Existing CUDA Projects to torchada

This guide shows how to migrate an existing PyTorch CUDA project to use torchada,
making it compatible with multiple GPU platforms.

## Quick Migration

### Step 1: Install torchada

```bash
pip install torchada
```

### Step 2: Add One Import

**Before:**
```python
import torch
from torch.utils.cpp_extension import CUDAExtension, BuildExtension, CUDA_HOME
```

**After:**
```python
import torchada  # Just add this one line!
import torch
from torch.utils.cpp_extension import CUDAExtension, BuildExtension, CUDA_HOME
```

That's it! **No other code changes needed.** Your existing `torch.cuda.*` code
and `torch.utils.cpp_extension` imports work on all supported platforms.

## Detailed Migration Examples

### Example 1: Basic GPU Usage

**Before:**
```python
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    tensor = torch.randn(100, 100).cuda()
    model = MyModel().cuda()
```

**After:**
```python
import torchada  # Add this line at the top - that's it!
import torch

# Rest of the code stays exactly the same!
if torch.cuda.is_available():
    device = torch.device("cuda")
    tensor = torch.randn(100, 100).cuda()
    model = MyModel().cuda()
```

### Example 2: torch.cuda APIs

All standard `torch.cuda` APIs work after importing torchada:

```python
import torchada
import torch

if torch.cuda.is_available():
    torch.cuda.set_device(0)
    print(f"Using: {torch.cuda.get_device_name()}")
    print(f"Memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    torch.cuda.synchronize()
```

### Example 3: Mixed Precision Training

**Before:**
```python
from torch.cuda.amp import autocast, GradScaler
```

**After:**
```python
import torchada  # Add this line
from torch.cuda.amp import autocast, GradScaler  # Same import works!
```

Or use the newer API:
```python
import torchada
import torch

with torch.amp.autocast(device_type='cuda'):
    output = model(input)
```

### Example 4: Building Extensions (setup.py)

**Before:**
```python
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension, CUDA_HOME

setup(
    name="my_extension",
    ext_modules=[
        CUDAExtension(
            name="my_extension",
            sources=["my_extension.cpp", "my_kernel.cu"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
```

**After:**
```python
import torchada  # Add this line at the top
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension, CUDA_HOME

# Exactly the same setup code works on all supported platforms!
setup(
    name="my_extension",
    ext_modules=[
        CUDAExtension(
            name="my_extension",
            sources=["my_extension.cpp", "my_kernel.cu"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
```

### Example 5: Distributed Training

**Before:**
```python
import torch.distributed as dist
dist.init_process_group(backend='nccl')
```

**After:**
```python
import torchada  # Add this line
import torch.distributed as dist
dist.init_process_group(backend='nccl')  # Works on all supported platforms
```

### Example 6: CUDA Graphs

**Before:**
```python
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    y = model(x)
```

**After:**
```python
import torchada  # Add this line
import torch

g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    y = model(x)
```

## What Happens Under the Hood

When you import torchada, it:

1. **Detects the platform**: Identifies the available GPU hardware
2. **Patches PyTorch modules**: Makes `torch.cuda` and `torch.utils.cpp_extension` work transparently
3. **Translates device strings**: `"cuda"` device strings work on any supported platform
4. **Maps backends**: `"nccl"` backend works on all supported platforms
5. **Converts symbols**: CUDA API calls in extensions are mapped to platform equivalents
6. **Handles compilation**: `.cu` files are compiled with the appropriate compiler

## Environment Variables

You can force a specific platform:

```bash
export TORCHADA_PLATFORM=cuda  # or cpu
```

## Common Patterns in Popular Projects

### vLLM-style setup.py

```python
# Just add this at the top of setup.py
import torchada

# Keep all your existing imports unchanged
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
```

### SGLang-style code

```python
# Just add this at the top
import torchada

# All existing torch.cuda code works unchanged
import torch.cuda
torch.cuda.synchronize()
```

## Tips for Migration

1. **Import torchada first**: Always import torchada before torch to ensure patches are applied
2. **Keep standard imports**: Use `from torch.utils.cpp_extension import ...` (not from torchada)
3. **Keep "cuda" strings**: No need to change device strings - torchada handles platform differences
4. **Test your code**: Verify your code works correctly after adding the torchada import

