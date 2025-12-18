# Migrating Existing CUDA Projects to torchada

This guide shows how to migrate an existing PyTorch CUDA project to use torchada,
making it compatible with both NVIDIA (CUDA) and Moore Threads (MUSA) GPUs.

## Quick Migration

### Step 1: Install torchada

```bash
pip install torchada
```

### Step 2: Update Imports

**Before (CUDA only):**
```python
import torch
import torch.cuda as cuda
from torch.utils.cpp_extension import CUDAExtension, BuildExtension, CUDA_HOME
```

**After (CUDA + MUSA):**
```python
import torchada  # Must import first to apply patches
import torch
from torchada import cuda
from torchada.utils.cpp_extension import CUDAExtension, BuildExtension, CUDA_HOME
```

That's it! Your code should now work on both platforms.

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
import torchada  # Add this line at the top
import torch

# Rest of the code stays exactly the same!
if torch.cuda.is_available():
    device = torch.device("cuda")
    tensor = torch.randn(100, 100).cuda()
    model = MyModel().cuda()
```

### Example 2: Using torchada.cuda Module

If you prefer explicit imports:

```python
from torchada import cuda

if cuda.is_available():
    cuda.set_device(0)
    print(f"Using: {cuda.get_device_name()}")
    print(f"Memory: {cuda.memory_allocated() / 1024**2:.2f} MB")
```

### Example 3: Mixed Precision Training

**Before:**
```python
from torch.cuda.amp import autocast, GradScaler
```

**After:**
```python
from torchada.cuda.amp import autocast, GradScaler
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
from setuptools import setup
from torchada.utils.cpp_extension import CUDAExtension, BuildExtension, CUDA_HOME

# Exactly the same setup code works on both CUDA and MUSA!
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

## What Happens Under the Hood

When you import torchada on a MUSA platform:

1. **Platform Detection**: torchada detects Moore Threads GPU
2. **Automatic Patching**: `tensor.cuda()` and `.to("cuda")` are patched to use MUSA
3. **Symbol Mapping**: CUDA API calls in extensions are mapped to MUSA equivalents
4. **Extension Building**: `.cu` files are compiled with MUSA compiler (mcc)

## Environment Variables

You can force a specific platform:

```bash
# Force MUSA platform
export TORCHADA_PLATFORM=musa

# Force CUDA platform  
export TORCHADA_PLATFORM=cuda

# Force CPU only
export TORCHADA_PLATFORM=cpu
```

## Common Patterns in Popular Projects

### vLLM-style imports

```python
# Original vLLM
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

# With torchada
from torchada.utils.cpp_extension import CUDAExtension, BuildExtension
```

### SGLang-style imports

```python
# Original SGLang
import torch.cuda

# With torchada
import torchada
import torch.cuda  # Still works thanks to patching!

# Or explicitly
from torchada import cuda
```

## Tips for Migration

1. **Import torchada first**: Always import torchada before torch to ensure patches are applied
2. **Keep "cuda" strings**: You don't need to change `"cuda"` to `"musa"` - torchada handles this
3. **Test on both platforms**: Verify your code works on both CUDA and MUSA if possible
4. **Check CUDA_HOME**: Use `from torchada.utils.cpp_extension import CUDA_HOME` for correct path

