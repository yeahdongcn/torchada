"""
Tests for platform detection and initialization.
"""

import pytest


class TestPlatformDetection:
    """Test platform detection functionality."""

    def test_import_torchada(self):
        """Test that torchada can be imported."""
        import torchada

        assert torchada is not None
        assert hasattr(torchada, "__version__")

    def test_detect_platform(self):
        """Test platform detection."""
        from torchada import Platform, detect_platform

        platform = detect_platform()
        assert platform in [Platform.CUDA, Platform.MUSA, Platform.CPU]

    def test_platform_detection_functions(self):
        """Test individual platform detection functions."""
        from torchada import (
            Platform,
            detect_platform,
            is_cpu_platform,
            is_cuda_platform,
            is_musa_platform,
        )

        platform = detect_platform()
        if platform == Platform.MUSA:
            assert is_musa_platform()
            assert not is_cuda_platform()
            assert not is_cpu_platform()
        elif platform == Platform.CUDA:
            assert not is_musa_platform()
            assert is_cuda_platform()
            assert not is_cpu_platform()
        else:
            assert not is_musa_platform()
            assert not is_cuda_platform()
            assert is_cpu_platform()

    def test_patches_applied(self):
        """Test that patches are automatically applied on import."""
        import torchada

        assert torchada.is_patched()

    def test_get_version(self):
        """Test version retrieval."""
        import torchada

        version = torchada.get_version()
        assert version == torchada.__version__
        assert isinstance(version, str)

    def test_get_platform(self):
        """Test get_platform helper."""
        import torchada

        platform = torchada.get_platform()
        assert platform == torchada.detect_platform()

    def test_get_backend(self):
        """Test get_backend returns a module."""
        import torchada

        backend = torchada.get_backend()
        assert backend is not None
        # Should have is_available method
        assert hasattr(backend, "is_available")
