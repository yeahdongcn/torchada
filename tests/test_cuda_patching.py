"""
Tests for torch.cuda patching functionality.

These tests verify that torch.cuda.* APIs work transparently on MUSA.
"""

import pytest
import sys


class TestTorchCudaModule:
    """Test torch.cuda module patching."""

    def test_torch_cuda_is_patched_on_musa(self):
        """Test that torch.cuda is patched to torch.musa on MUSA platform."""
        import torchada
        import torch

        if torchada.is_musa_platform():
            # torch.cuda should now be torch_musa
            assert "torch_musa" in torch.cuda.__name__ or "musa" in str(torch.cuda)
        else:
            # On CUDA/CPU, torch.cuda should remain unchanged
            assert "cuda" in torch.cuda.__name__

    def test_torch_cuda_is_available(self):
        """Test torch.cuda.is_available() works."""
        import torchada
        import torch

        result = torch.cuda.is_available()
        assert isinstance(result, bool)

        if torchada.is_musa_platform():
            # On MUSA platform, should return True if MUSA is available
            import torch_musa
            assert result == torch_musa.is_available()

    def test_torch_cuda_device_count(self):
        """Test torch.cuda.device_count() works."""
        import torchada
        import torch

        count = torch.cuda.device_count()
        assert isinstance(count, int)
        assert count >= 0

        if torchada.is_musa_platform() and torch.cuda.is_available():
            import torch_musa
            assert count == torch_musa.device_count()

    def test_torch_cuda_current_device(self):
        """Test torch.cuda.current_device() works when GPU available."""
        import torchada
        import torch

        if torch.cuda.is_available():
            device_id = torch.cuda.current_device()
            assert isinstance(device_id, int)
            assert device_id >= 0
            assert device_id < torch.cuda.device_count()

    def test_torch_cuda_get_device_name(self):
        """Test torch.cuda.get_device_name() works."""
        import torchada
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name()
            assert isinstance(name, str)
            assert len(name) > 0

            if torchada.is_musa_platform():
                # MUSA device names typically contain "MTT" or "Moore"
                # but this is hardware-dependent, so just check it's not empty
                pass

    def test_torch_cuda_synchronize(self):
        """Test torch.cuda.synchronize() works."""
        import torchada
        import torch

        if torch.cuda.is_available():
            # Should not raise
            torch.cuda.synchronize()

    def test_torch_cuda_empty_cache(self):
        """Test torch.cuda.empty_cache() works."""
        import torchada
        import torch

        if torch.cuda.is_available():
            # Should not raise
            torch.cuda.empty_cache()

    def test_torch_cuda_memory_allocated(self):
        """Test torch.cuda.memory_allocated() works."""
        import torchada
        import torch

        if torch.cuda.is_available():
            mem = torch.cuda.memory_allocated()
            assert isinstance(mem, int)
            assert mem >= 0

    def test_torch_cuda_memory_reserved(self):
        """Test torch.cuda.memory_reserved() works."""
        import torchada
        import torch

        if torch.cuda.is_available():
            mem = torch.cuda.memory_reserved()
            assert isinstance(mem, int)
            assert mem >= 0

    def test_torch_cuda_set_device(self):
        """Test torch.cuda.set_device() works."""
        import torchada
        import torch

        if torch.cuda.is_available():
            current = torch.cuda.current_device()
            torch.cuda.set_device(current)
            assert torch.cuda.current_device() == current


class TestTorchCudaAmp:
    """Test torch.cuda.amp module patching."""

    def test_import_autocast(self):
        """Test that autocast can be imported from torch.cuda.amp."""
        import torchada
        from torch.cuda.amp import autocast
        assert autocast is not None

    def test_import_grad_scaler(self):
        """Test that GradScaler can be imported from torch.cuda.amp."""
        import torchada
        from torch.cuda.amp import GradScaler
        assert GradScaler is not None

    def test_autocast_context_manager(self):
        """Test autocast works as context manager."""
        import torchada
        import torch
        from torch.cuda.amp import autocast

        if torch.cuda.is_available():
            try:
                with autocast():
                    x = torch.randn(2, 2, device="cuda")
                    assert x.device.type in ("cuda", "musa")
            except RuntimeError as e:
                # MUSA driver issue, not torchada
                if "MUSA" in str(e) or "invalid device function" in str(e):
                    pytest.skip(f"MUSA driver issue: {e}")
                raise

    def test_grad_scaler_creation(self):
        """Test GradScaler can be created."""
        import torchada
        import torch
        from torch.cuda.amp import GradScaler

        if torch.cuda.is_available():
            scaler = GradScaler()
            assert scaler is not None


class TestCUDAGraph:
    """Test CUDAGraph aliasing."""

    def test_cuda_graph_alias(self):
        """Test torch.cuda.CUDAGraph is available."""
        import torchada
        import torch

        assert hasattr(torch.cuda, 'CUDAGraph')
        assert hasattr(torch.cuda, 'MUSAGraph')

    def test_cuda_graph_is_musa_graph(self):
        """Test torch.cuda.CUDAGraph is aliased to MUSAGraph on MUSA."""
        import torchada
        import torch

        if torchada.is_musa_platform():
            assert torch.cuda.CUDAGraph is torch.cuda.MUSAGraph

    def test_cuda_graph_creation(self):
        """Test CUDAGraph can be created."""
        import torchada
        import torch

        if torch.cuda.is_available():
            try:
                g = torch.cuda.CUDAGraph()
                assert g is not None
            except RuntimeError as e:
                if "MUSA" in str(e) or "invalid device function" in str(e):
                    pytest.skip(f"MUSA driver issue: {e}")
                raise

    def test_graphs_module(self):
        """Test torch.cuda.graphs module is available."""
        import torchada
        import torch

        assert hasattr(torch.cuda, 'graphs')

    def test_make_graphed_callables(self):
        """Test make_graphed_callables is available."""
        import torchada
        import torch

        assert hasattr(torch.cuda, 'make_graphed_callables')

    def test_graph_pool_handle(self):
        """Test graph_pool_handle is available."""
        import torchada
        import torch

        assert hasattr(torch.cuda, 'graph_pool_handle')


class TestDistributedBackend:
    """Test distributed backend patching."""

    def test_original_init_available(self):
        """Test original init_process_group is saved."""
        import torchada

        if torchada.is_musa_platform():
            original = torchada.get_original_init_process_group()
            assert original is not None

    def test_mccl_backend_available(self):
        """Test MCCL backend is registered."""
        import torchada
        import torch.distributed as dist

        if torchada.is_musa_platform():
            assert hasattr(dist.Backend, 'MCCL')

    def test_nccl_backend_available(self):
        """Test NCCL backend constant is still available."""
        import torchada
        import torch.distributed as dist

        assert hasattr(dist.Backend, 'NCCL')


class TestNCCLModule:
    """Test NCCL to MCCL module patching."""

    def test_mccl_module_available(self):
        """Test torch.cuda.mccl is available."""
        import torchada
        import torch

        if torchada.is_musa_platform():
            assert hasattr(torch.cuda, 'mccl')


class TestRNGFunctions:
    """Test RNG functions are available through torch.cuda."""

    def test_get_rng_state(self):
        """Test torch.cuda.get_rng_state is available."""
        import torchada
        import torch

        assert hasattr(torch.cuda, 'get_rng_state')

    def test_set_rng_state(self):
        """Test torch.cuda.set_rng_state is available."""
        import torchada
        import torch

        assert hasattr(torch.cuda, 'set_rng_state')

    def test_manual_seed(self):
        """Test torch.cuda.manual_seed is available."""
        import torchada
        import torch

        assert hasattr(torch.cuda, 'manual_seed')

    def test_manual_seed_all(self):
        """Test torch.cuda.manual_seed_all is available."""
        import torchada
        import torch

        assert hasattr(torch.cuda, 'manual_seed_all')

    def test_seed(self):
        """Test torch.cuda.seed is available."""
        import torchada
        import torch

        assert hasattr(torch.cuda, 'seed')

    def test_initial_seed(self):
        """Test torch.cuda.initial_seed is available."""
        import torchada
        import torch

        assert hasattr(torch.cuda, 'initial_seed')

    def test_manual_seed_works(self):
        """Test torch.cuda.manual_seed actually works."""
        import torchada
        import torch

        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)


class TestMemoryFunctions:
    """Test additional memory functions."""

    def test_max_memory_allocated(self):
        """Test torch.cuda.max_memory_allocated is available."""
        import torchada
        import torch

        assert hasattr(torch.cuda, 'max_memory_allocated')

    def test_max_memory_reserved(self):
        """Test torch.cuda.max_memory_reserved is available."""
        import torchada
        import torch

        assert hasattr(torch.cuda, 'max_memory_reserved')

    def test_memory_stats(self):
        """Test torch.cuda.memory_stats is available."""
        import torchada
        import torch

        assert hasattr(torch.cuda, 'memory_stats')

    def test_memory_summary(self):
        """Test torch.cuda.memory_summary is available."""
        import torchada
        import torch

        assert hasattr(torch.cuda, 'memory_summary')

    def test_memory_snapshot(self):
        """Test torch.cuda.memory_snapshot is available."""
        import torchada
        import torch

        assert hasattr(torch.cuda, 'memory_snapshot')

    def test_reset_peak_memory_stats(self):
        """Test torch.cuda.reset_peak_memory_stats is available."""
        import torchada
        import torch

        assert hasattr(torch.cuda, 'reset_peak_memory_stats')

    def test_mem_get_info(self):
        """Test torch.cuda.mem_get_info is available."""
        import torchada
        import torch

        assert hasattr(torch.cuda, 'mem_get_info')

        if torch.cuda.is_available():
            try:
                free, total = torch.cuda.mem_get_info()
                assert free >= 0
                assert total > 0
            except RuntimeError as e:
                if "MUSA" in str(e):
                    pytest.skip(f"MUSA driver issue: {e}")
                raise


class TestStreamAndEvent:
    """Test Stream and Event classes."""

    def test_stream_class(self):
        """Test torch.cuda.Stream is available."""
        import torchada
        import torch

        assert hasattr(torch.cuda, 'Stream')

    def test_event_class(self):
        """Test torch.cuda.Event is available."""
        import torchada
        import torch

        assert hasattr(torch.cuda, 'Event')

    def test_external_stream_class(self):
        """Test torch.cuda.ExternalStream is available."""
        import torchada
        import torch

        assert hasattr(torch.cuda, 'ExternalStream')

    def test_current_stream(self):
        """Test torch.cuda.current_stream is available."""
        import torchada
        import torch

        assert hasattr(torch.cuda, 'current_stream')

    def test_default_stream(self):
        """Test torch.cuda.default_stream is available."""
        import torchada
        import torch

        assert hasattr(torch.cuda, 'default_stream')

    def test_set_stream(self):
        """Test torch.cuda.set_stream is available."""
        import torchada
        import torch

        assert hasattr(torch.cuda, 'set_stream')

    def test_stream_context_manager(self):
        """Test torch.cuda.stream is available."""
        import torchada
        import torch

        assert hasattr(torch.cuda, 'stream')


class TestContextManagers:
    """Test device context managers."""

    def test_device_context_manager(self):
        """Test torch.cuda.device is available."""
        import torchada
        import torch

        assert hasattr(torch.cuda, 'device')

    def test_device_of_context_manager(self):
        """Test torch.cuda.device_of is available."""
        import torchada
        import torch

        assert hasattr(torch.cuda, 'device_of')


class TestDeviceFunctions:
    """Test additional device functions."""

    def test_get_device_properties(self):
        """Test torch.cuda.get_device_properties is available."""
        import torchada
        import torch

        assert hasattr(torch.cuda, 'get_device_properties')

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            assert props is not None
            assert hasattr(props, 'name')
            assert hasattr(props, 'total_memory')

    def test_get_device_capability(self):
        """Test torch.cuda.get_device_capability is available."""
        import torchada
        import torch

        assert hasattr(torch.cuda, 'get_device_capability')

        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability(0)
            assert isinstance(cap, tuple)
            assert len(cap) == 2

    def test_is_initialized(self):
        """Test torch.cuda.is_initialized is available."""
        import torchada
        import torch

        assert hasattr(torch.cuda, 'is_initialized')


class TestTorchVersionCuda:
    """Test torch.version.cuda patching."""

    def test_torch_version_cuda_is_set(self):
        """Test torch.version.cuda is set on MUSA platform."""
        import torchada
        import torch

        if torchada.is_musa_platform():
            assert torch.version.cuda is not None
            assert isinstance(torch.version.cuda, str)

    def test_torch_version_cuda_matches_musa(self):
        """Test torch.version.cuda matches torch.version.musa on MUSA platform."""
        import torchada
        import torch

        if torchada.is_musa_platform():
            assert torch.version.cuda == str(torch.version.musa)


class TestTensorIsCuda:
    """Test tensor.is_cuda patching."""

    def test_cpu_tensor_is_cuda_false(self):
        """Test CPU tensor.is_cuda returns False."""
        import torchada
        import torch

        cpu_tensor = torch.empty(10, 10)
        assert cpu_tensor.is_cuda is False

    def test_musa_tensor_is_cuda_true(self):
        """Test MUSA tensor.is_cuda returns True on MUSA platform."""
        import torchada
        import torch

        if torchada.is_musa_platform() and torch.cuda.is_available():
            try:
                musa_tensor = torch.empty(10, 10, device='cuda:0')
                assert musa_tensor.is_cuda is True
                assert musa_tensor.is_musa is True
            except RuntimeError as e:
                if "MUSA" in str(e) or "invalid device function" in str(e):
                    pytest.skip(f"MUSA driver issue: {e}")
                raise


class TestNvtxStub:
    """Test torch.cuda.nvtx stub module."""

    def test_nvtx_module_available(self):
        """Test torch.cuda.nvtx is available."""
        import torchada
        import torch

        assert hasattr(torch.cuda, 'nvtx')

    def test_nvtx_mark(self):
        """Test torch.cuda.nvtx.mark is available and callable."""
        import torchada
        import torch.cuda.nvtx as nvtx

        assert hasattr(nvtx, 'mark')
        # Should not raise
        nvtx.mark("test")

    def test_nvtx_range_push_pop(self):
        """Test torch.cuda.nvtx.range_push and range_pop are available."""
        import torchada
        import torch.cuda.nvtx as nvtx

        assert hasattr(nvtx, 'range_push')
        assert hasattr(nvtx, 'range_pop')
        # Should not raise
        nvtx.range_push("test")
        nvtx.range_pop()

    def test_nvtx_range_context_manager(self):
        """Test torch.cuda.nvtx.range context manager works."""
        import torchada
        import torch.cuda.nvtx as nvtx

        assert hasattr(nvtx, 'range')
        # Should not raise
        with nvtx.range("test"):
            pass
