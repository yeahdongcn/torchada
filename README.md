# torchada

**Adapter package for torch_musa to act exactly like PyTorch CUDA**

torchada provides a unified interface that works transparently on both NVIDIA GPUs (CUDA) and Moore Threads GPUs (MUSA). Write your code once using CUDA APIs, and it will run on MUSA hardware without any changes.

## Features

- **Automatic Platform Detection**: Detects whether you're running on CUDA or MUSA
- **Drop-in Replacement**: Use the same CUDA APIs on MUSA hardware
- **Transparent Device Mapping**: `tensor.cuda()` and `tensor.to("cuda")` work on MUSA
- **Extension Building**: `CUDAExtension` and `BuildExtension` work on both platforms
- **Source Code Porting**: Automatic CUDA → MUSA symbol mapping for C++/CUDA extensions

## Installation

```bash
pip install torchada

# Or install from source
git clone https://github.com/yeahdongcn/torchada.git
cd torchada
pip install -e .
```

## Quick Start

### Basic Usage

```python
import torchada  # Automatically patches PyTorch for MUSA compatibility
import torch

# These work on both CUDA and MUSA platforms:
if torch.cuda.is_available():
    device = torch.device("cuda")
    tensor = torch.randn(10, 10).cuda()
    model = MyModel().cuda()
```

### Using torchada.cuda

```python
from torchada import cuda

# Works on both CUDA and MUSA
if cuda.is_available():
    print(f"Device count: {cuda.device_count()}")
    print(f"Current device: {cuda.current_device()}")
    print(f"Device name: {cuda.get_device_name()}")

    cuda.set_device(0)
    cuda.synchronize()
```

### Building C++ Extensions

```python
# setup.py
from setuptools import setup
from torchada.utils.cpp_extension import CUDAExtension, BuildExtension, CUDA_HOME

print(f"Building with CUDA/MUSA home: {CUDA_HOME}")

ext_modules = [
    CUDAExtension(
        name="my_extension",
        sources=[
            "my_extension.cpp",
            "my_extension_kernel.cu",
        ],
        extra_compile_args={
            "cxx": ["-O3"],
            "nvcc": ["-O3"],  # Automatically mapped to mcc on MUSA
        },
    ),
]

setup(
    name="my_package",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)
```

### JIT Compilation

```python
from torchada.utils.cpp_extension import load

# Load extension at runtime (works on both CUDA and MUSA)
my_extension = load(
    name="my_extension",
    sources=["my_extension.cpp", "my_extension_kernel.cu"],
    verbose=True,
)
```

### Mixed Precision Training

```python
from torchada.cuda.amp import autocast, GradScaler

model = MyModel().cuda()
optimizer = torch.optim.Adam(model.parameters())
scaler = GradScaler()

for data, target in dataloader:
    data, target = data.cuda(), target.cuda()

    with autocast():
        output = model(data)
        loss = criterion(output, target)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

## Platform Detection

torchada automatically detects the platform. You can also force a specific platform:

```python
# Force specific platform via environment variable
export TORCHADA_PLATFORM=musa  # or cuda, cpu

# Or check programmatically
from torchada import detect_platform, Platform

platform = detect_platform()
if platform == Platform.MUSA:
    print("Running on Moore Threads GPU")
elif platform == Platform.CUDA:
    print("Running on NVIDIA GPU")
```

## API Reference

### torchada

| Function | Description |
|----------|-------------|
| `detect_platform()` | Returns the detected platform (CUDA, MUSA, or CPU) |
| `is_musa_platform()` | Check if running on MUSA |
| `is_cuda_platform()` | Check if running on CUDA |
| `get_device_name()` | Get device name string ("cuda", "musa", or "cpu") |

### torchada.cuda

Same API as `torch.cuda`, including:
- `is_available()`, `device_count()`, `current_device()`, `set_device()`
- `memory_allocated()`, `memory_reserved()`, `empty_cache()`
- `synchronize()`, `Stream`, `Event`

### torchada.utils.cpp_extension

| Symbol | Description |
|--------|-------------|
| `CUDAExtension` | Creates CUDA or MUSA extension based on platform |
| `CppExtension` | Creates C++ extension (no GPU code) |
| `BuildExtension` | Build command for extensions |
| `CUDA_HOME` | Path to CUDA/MUSA installation |
| `load()` | JIT compile and load extension |
| `include_paths()` | Get include paths |
| `library_paths()` | Get library paths |

## Symbol Mapping

torchada automatically maps CUDA symbols to MUSA equivalents when building extensions:

| CUDA | MUSA |
|------|------|
| `cudaMalloc` | `musaMalloc` |
| `cudaMemcpy` | `musaMemcpy` |
| `cudaStream_t` | `musaStream_t` |
| `cublasHandle_t` | `mublasHandle_t` |
| `curandState` | `murandState` |
| `at::cuda` | `at::musa` |
| `c10::cuda` | `c10::musa` |
| ... | ... |

See `src/torchada/_mapping.py` for the complete mapping table.

## License

MIT License