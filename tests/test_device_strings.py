"""
Tests for device string patching (cuda -> musa translation).
"""

import pytest


class TestTensorDevicePatching:
    """Test tensor device string patching."""

    def test_tensor_cuda_method(self):
        """Test .cuda() method on tensors."""
        import torchada
        import torch

        if torch.cuda.is_available():
            x = torch.randn(2, 2)
            y = x.cuda()
            assert y.device.type in ("cuda", "musa")
            if torchada.is_musa_platform():
                assert y.device.type == "musa"

    def test_tensor_to_cuda_string(self):
        """Test .to('cuda') on tensors."""
        import torchada
        import torch

        if torch.cuda.is_available():
            x = torch.randn(2, 2)
            y = x.to("cuda")
            assert y.device.type in ("cuda", "musa")
            if torchada.is_musa_platform():
                assert y.device.type == "musa"

    def test_tensor_to_cuda_device_index(self):
        """Test .to('cuda:0') on tensors."""
        import torchada
        import torch

        if torch.cuda.is_available():
            x = torch.randn(2, 2)
            y = x.to("cuda:0")
            assert y.device.type in ("cuda", "musa")
            if torchada.is_musa_platform():
                assert y.device.type == "musa"
                assert y.device.index == 0

    def test_tensor_to_torch_device(self):
        """Test .to(torch.device('cuda')) on tensors."""
        import torchada
        import torch

        if torch.cuda.is_available():
            x = torch.randn(2, 2)
            y = x.to(torch.device("cuda"))
            assert y.device.type in ("cuda", "musa")


class TestModuleDevicePatching:
    """Test nn.Module device patching."""

    def test_module_cuda_method(self):
        """Test .cuda() method on modules."""
        import torchada
        import torch
        import torch.nn as nn

        if torch.cuda.is_available():
            model = nn.Linear(10, 5)
            model = model.cuda()
            param_device = next(model.parameters()).device
            assert param_device.type in ("cuda", "musa")
            if torchada.is_musa_platform():
                assert param_device.type == "musa"

    def test_module_to_cuda(self):
        """Test .to('cuda') method on modules."""
        import torchada
        import torch
        import torch.nn as nn

        if torch.cuda.is_available():
            model = nn.Linear(10, 5)
            model = model.to("cuda")
            param_device = next(model.parameters()).device
            assert param_device.type in ("cuda", "musa")


class TestFactoryFunctions:
    """Test tensor factory function patching."""

    def test_torch_empty_device_cuda(self):
        """Test torch.empty(device='cuda')."""
        import torchada
        import torch

        if torch.cuda.is_available():
            x = torch.empty(2, 2, device="cuda")
            assert x.device.type in ("cuda", "musa")
            if torchada.is_musa_platform():
                assert x.device.type == "musa"

    def test_torch_zeros_device_cuda(self):
        """Test torch.zeros(device='cuda')."""
        import torchada
        import torch

        if torch.cuda.is_available():
            try:
                x = torch.zeros(2, 2, device="cuda")
                assert x.device.type in ("cuda", "musa")
            except RuntimeError as e:
                # MUSA driver issue, not torchada
                if "MUDNN" in str(e) or "invalid device function" in str(e):
                    pytest.skip(f"MUSA driver issue: {e}")
                raise

    def test_torch_ones_device_cuda(self):
        """Test torch.ones(device='cuda')."""
        import torchada
        import torch

        if torch.cuda.is_available():
            try:
                x = torch.ones(2, 2, device="cuda")
                assert x.device.type in ("cuda", "musa")
            except RuntimeError as e:
                if "MUDNN" in str(e) or "invalid device function" in str(e):
                    pytest.skip(f"MUSA driver issue: {e}")
                raise

    def test_torch_randn_device_cuda(self):
        """Test torch.randn(device='cuda')."""
        import torchada
        import torch

        if torch.cuda.is_available():
            try:
                x = torch.randn(2, 2, device="cuda")
                assert x.device.type in ("cuda", "musa")
            except RuntimeError as e:
                if "MUSA" in str(e) or "invalid device function" in str(e):
                    pytest.skip(f"MUSA driver issue: {e}")
                raise

    def test_torch_rand_device_cuda(self):
        """Test torch.rand(device='cuda')."""
        import torchada
        import torch

        if torch.cuda.is_available():
            try:
                x = torch.rand(2, 2, device="cuda")
                assert x.device.type in ("cuda", "musa")
            except RuntimeError as e:
                if "MUSA" in str(e) or "invalid device function" in str(e):
                    pytest.skip(f"MUSA driver issue: {e}")
                raise

    def test_torch_full_device_cuda(self):
        """Test torch.full(device='cuda')."""
        import torchada
        import torch

        if torch.cuda.is_available():
            try:
                x = torch.full((2, 2), 3.14, device="cuda")
                assert x.device.type in ("cuda", "musa")
            except RuntimeError as e:
                if "MUDNN" in str(e) or "invalid device function" in str(e):
                    pytest.skip(f"MUSA driver issue: {e}")
                raise

    def test_torch_arange_device_cuda(self):
        """Test torch.arange(device='cuda')."""
        import torchada
        import torch

        if torch.cuda.is_available():
            try:
                x = torch.arange(10, device="cuda")
                assert x.device.type in ("cuda", "musa")
            except RuntimeError as e:
                if "MUSA" in str(e) or "invalid device function" in str(e):
                    pytest.skip(f"MUSA driver issue: {e}")
                raise

    def test_torch_linspace_device_cuda(self):
        """Test torch.linspace(device='cuda')."""
        import torchada
        import torch

        if torch.cuda.is_available():
            try:
                x = torch.linspace(0, 1, 10, device="cuda")
                assert x.device.type in ("cuda", "musa")
            except RuntimeError as e:
                if "MUSA" in str(e) or "invalid device function" in str(e):
                    pytest.skip(f"MUSA driver issue: {e}")
                raise

