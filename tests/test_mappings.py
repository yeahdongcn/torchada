"""
Tests for CUDA to MUSA mapping rules.
"""

import pytest


class TestMappingImports:
    """Test mapping module imports."""

    def test_import_mapping_rule(self):
        """Test _MAPPING_RULE can be imported."""
        from torchada._mapping import _MAPPING_RULE
        assert isinstance(_MAPPING_RULE, dict)
        assert len(_MAPPING_RULE) > 0

    def test_import_ext_replaced_mapping(self):
        """Test EXT_REPLACED_MAPPING can be imported."""
        from torchada._mapping import EXT_REPLACED_MAPPING
        assert isinstance(EXT_REPLACED_MAPPING, dict)


class TestATenMappings:
    """Test ATen CUDA to MUSA mappings."""

    def test_aten_cuda_namespace(self):
        from torchada._mapping import _MAPPING_RULE
        assert _MAPPING_RULE["at::cuda"] == "at::musa"
        assert _MAPPING_RULE["at::cuda::"] == "at::musa::"

    def test_aten_cuda_includes(self):
        from torchada._mapping import _MAPPING_RULE
        assert _MAPPING_RULE["ATen/cuda"] == "ATen/musa"
        assert _MAPPING_RULE["ATen/CUDAContext.h"] == "ATen/musa/MUSAContext.h"
        assert _MAPPING_RULE["ATen/CUDAGeneratorImpl.h"] == "ATen/musa/MUSAGeneratorImpl.h"


class TestC10Mappings:
    """Test C10 CUDA to MUSA mappings."""

    def test_c10_cuda_namespace(self):
        from torchada._mapping import _MAPPING_RULE
        assert _MAPPING_RULE["c10::cuda"] == "c10::musa"
        assert _MAPPING_RULE["c10::cuda::"] == "c10::musa::"
        assert _MAPPING_RULE["c10/cuda"] == "c10/musa"

    def test_c10_device_type(self):
        from torchada._mapping import _MAPPING_RULE
        assert _MAPPING_RULE["c10::DeviceType::CUDA"] == "c10::DeviceType::MUSA"


class TestTorchMappings:
    """Test torch namespace mappings."""

    def test_torch_cuda_namespace(self):
        from torchada._mapping import _MAPPING_RULE
        assert _MAPPING_RULE["torch::cuda"] == "torch::musa"
        assert _MAPPING_RULE["torch.cuda"] == "torch.musa"

    def test_torch_device_type(self):
        from torchada._mapping import _MAPPING_RULE
        assert _MAPPING_RULE["at::kCUDA"] == "at::kMUSA"
        assert _MAPPING_RULE["at::DeviceType::CUDA"] == "at::DeviceType::MUSA"
        assert _MAPPING_RULE["torch::kCUDA"] == "torch::kMUSA"


class TestCuBLASMappings:
    """Test cuBLAS to muBLAS mappings."""

    def test_cublas_basic(self):
        from torchada._mapping import _MAPPING_RULE
        assert _MAPPING_RULE["cublas"] == "mublas"
        assert _MAPPING_RULE["CUBLAS"] == "MUBLAS"

    def test_cublas_types(self):
        from torchada._mapping import _MAPPING_RULE
        assert _MAPPING_RULE["cublasHandle_t"] == "mublasHandle_t"
        assert _MAPPING_RULE["cublasStatus_t"] == "mublasStatus_t"

    def test_cublas_functions(self):
        from torchada._mapping import _MAPPING_RULE
        assert _MAPPING_RULE["cublasCreate"] == "mublasCreate"
        assert _MAPPING_RULE["cublasDestroy"] == "mublasDestroy"
        assert _MAPPING_RULE["cublasSetStream"] == "mublasSetStream"
        assert _MAPPING_RULE["cublasGetStream"] == "mublasGetStream"

    def test_cublas_gemm(self):
        from torchada._mapping import _MAPPING_RULE
        assert _MAPPING_RULE["cublasSgemm"] == "mublasSgemm"
        assert _MAPPING_RULE["cublasDgemm"] == "mublasDgemm"
        assert _MAPPING_RULE["cublasHgemm"] == "mublasHgemm"
        assert _MAPPING_RULE["cublasGemmEx"] == "mublasGemmEx"

    def test_cublas_batched(self):
        from torchada._mapping import _MAPPING_RULE
        assert _MAPPING_RULE["cublasGemmBatchedEx"] == "mublasGemmBatchedEx"
        assert _MAPPING_RULE["cublasGemmStridedBatchedEx"] == "mublasGemmStridedBatchedEx"
        assert _MAPPING_RULE["cublasSgemmBatched"] == "mublasSgemmBatched"
        assert _MAPPING_RULE["cublasDgemmBatched"] == "mublasDgemmBatched"

    def test_cublaslt(self):
        from torchada._mapping import _MAPPING_RULE
        assert _MAPPING_RULE["cublasLtCreate"] == "mublasLtCreate"
        assert _MAPPING_RULE["cublasLtDestroy"] == "mublasLtDestroy"
        assert _MAPPING_RULE["cublasLtHandle_t"] == "mublasLtHandle_t"
        assert _MAPPING_RULE["cublasLtMatmul"] == "mublasLtMatmul"


class TestCuRANDMappings:
    """Test cuRAND to muRAND mappings."""

    def test_curand_basic(self):
        from torchada._mapping import _MAPPING_RULE
        assert _MAPPING_RULE["curand"] == "murand"
        assert _MAPPING_RULE["CURAND"] == "MURAND"

    def test_curand_types(self):
        from torchada._mapping import _MAPPING_RULE
        assert _MAPPING_RULE["curandState"] == "murandState"
        assert _MAPPING_RULE["curandStatePhilox4_32_10_t"] == "murandStatePhilox4_32_10_t"

    def test_curand_functions(self):
        from torchada._mapping import _MAPPING_RULE
        assert _MAPPING_RULE["curand_init"] == "murand_init"
        assert _MAPPING_RULE["curand_uniform"] == "murand_uniform"
        assert _MAPPING_RULE["curand_uniform4"] == "murand_uniform4"
        assert _MAPPING_RULE["curand_normal"] == "murand_normal"
        assert _MAPPING_RULE["curand_normal4"] == "murand_normal4"


class TestCuDNNMappings:
    """Test cuDNN to muDNN mappings."""

    def test_cudnn_basic(self):
        from torchada._mapping import _MAPPING_RULE
        assert _MAPPING_RULE["cudnn"] == "mudnn"
        assert _MAPPING_RULE["CUDNN"] == "MUDNN"

    def test_cudnn_types(self):
        from torchada._mapping import _MAPPING_RULE
        assert _MAPPING_RULE["cudnnHandle_t"] == "mudnnHandle_t"
        assert _MAPPING_RULE["cudnnStatus_t"] == "mudnnStatus_t"

    def test_cudnn_functions(self):
        from torchada._mapping import _MAPPING_RULE
        assert _MAPPING_RULE["cudnnCreate"] == "mudnnCreate"
        assert _MAPPING_RULE["cudnnDestroy"] == "mudnnDestroy"
        assert _MAPPING_RULE["cudnnSetStream"] == "mudnnSetStream"


class TestCUDARuntimeMappings:
    """Test CUDA runtime to MUSA runtime mappings."""

    def test_memory_functions(self):
        from torchada._mapping import _MAPPING_RULE
        assert _MAPPING_RULE["cudaMalloc"] == "musaMalloc"
        assert _MAPPING_RULE["cudaFree"] == "musaFree"
        assert _MAPPING_RULE["cudaMemcpy"] == "musaMemcpy"
        assert _MAPPING_RULE["cudaMemcpyAsync"] == "musaMemcpyAsync"
        assert _MAPPING_RULE["cudaMemset"] == "musaMemset"
        assert _MAPPING_RULE["cudaMemsetAsync"] == "musaMemsetAsync"

    def test_host_memory_functions(self):
        from torchada._mapping import _MAPPING_RULE
        assert _MAPPING_RULE["cudaHostAlloc"] == "musaHostAlloc"
        assert _MAPPING_RULE["cudaHostFree"] == "musaHostFree"
        assert _MAPPING_RULE["cudaMallocHost"] == "musaMallocHost"
        assert _MAPPING_RULE["cudaFreeHost"] == "musaFreeHost"
        assert _MAPPING_RULE["cudaMallocManaged"] == "musaMallocManaged"

    def test_async_memory_functions(self):
        from torchada._mapping import _MAPPING_RULE
        assert _MAPPING_RULE["cudaMallocAsync"] == "musaMallocAsync"
        assert _MAPPING_RULE["cudaFreeAsync"] == "musaFreeAsync"
        assert _MAPPING_RULE["cudaMemcpy2D"] == "musaMemcpy2D"
        assert _MAPPING_RULE["cudaMemcpy2DAsync"] == "musaMemcpy2DAsync"
        assert _MAPPING_RULE["cudaMemcpy3D"] == "musaMemcpy3D"
        assert _MAPPING_RULE["cudaMemcpy3DAsync"] == "musaMemcpy3DAsync"

    def test_device_functions(self):
        from torchada._mapping import _MAPPING_RULE
        assert _MAPPING_RULE["cudaDeviceSynchronize"] == "musaDeviceSynchronize"
        assert _MAPPING_RULE["cudaGetDevice"] == "musaGetDevice"
        assert _MAPPING_RULE["cudaSetDevice"] == "musaSetDevice"
        assert _MAPPING_RULE["cudaGetDeviceCount"] == "musaGetDeviceCount"
        assert _MAPPING_RULE["cudaGetDeviceProperties"] == "musaGetDeviceProperties"
        assert _MAPPING_RULE["cudaDeviceGetAttribute"] == "musaDeviceGetAttribute"

    def test_memcpy_kinds(self):
        from torchada._mapping import _MAPPING_RULE
        assert _MAPPING_RULE["cudaMemcpyHostToDevice"] == "musaMemcpyHostToDevice"
        assert _MAPPING_RULE["cudaMemcpyDeviceToHost"] == "musaMemcpyDeviceToHost"
        assert _MAPPING_RULE["cudaMemcpyDeviceToDevice"] == "musaMemcpyDeviceToDevice"
        assert _MAPPING_RULE["cudaMemcpyHostToHost"] == "musaMemcpyHostToHost"


class TestStreamEventMappings:
    """Test CUDA stream/event mappings."""

    def test_stream_types(self):
        from torchada._mapping import _MAPPING_RULE
        assert _MAPPING_RULE["cudaStream_t"] == "musaStream_t"
        assert _MAPPING_RULE["cudaEvent_t"] == "musaEvent_t"

    def test_stream_functions(self):
        from torchada._mapping import _MAPPING_RULE
        assert _MAPPING_RULE["cudaStreamCreate"] == "musaStreamCreate"
        assert _MAPPING_RULE["cudaStreamDestroy"] == "musaStreamDestroy"
        assert _MAPPING_RULE["cudaStreamSynchronize"] == "musaStreamSynchronize"
        assert _MAPPING_RULE["cudaStreamQuery"] == "musaStreamQuery"
        assert _MAPPING_RULE["cudaStreamWaitEvent"] == "musaStreamWaitEvent"

    def test_stream_flags(self):
        from torchada._mapping import _MAPPING_RULE
        assert _MAPPING_RULE["cudaStreamDefault"] == "musaStreamDefault"
        assert _MAPPING_RULE["cudaStreamNonBlocking"] == "musaStreamNonBlocking"
        assert _MAPPING_RULE["cudaStreamCreateWithFlags"] == "musaStreamCreateWithFlags"
        assert _MAPPING_RULE["cudaStreamCreateWithPriority"] == "musaStreamCreateWithPriority"

    def test_event_functions(self):
        from torchada._mapping import _MAPPING_RULE
        assert _MAPPING_RULE["cudaEventCreate"] == "musaEventCreate"
        assert _MAPPING_RULE["cudaEventDestroy"] == "musaEventDestroy"
        assert _MAPPING_RULE["cudaEventRecord"] == "musaEventRecord"
        assert _MAPPING_RULE["cudaEventSynchronize"] == "musaEventSynchronize"
        assert _MAPPING_RULE["cudaEventElapsedTime"] == "musaEventElapsedTime"
        assert _MAPPING_RULE["cudaEventQuery"] == "musaEventQuery"

    def test_event_flags(self):
        from torchada._mapping import _MAPPING_RULE
        assert _MAPPING_RULE["cudaEventDefault"] == "musaEventDefault"
        assert _MAPPING_RULE["cudaEventBlockingSync"] == "musaEventBlockingSync"
        assert _MAPPING_RULE["cudaEventDisableTiming"] == "musaEventDisableTiming"
        assert _MAPPING_RULE["cudaEventCreateWithFlags"] == "musaEventCreateWithFlags"


class TestErrorHandlingMappings:
    """Test CUDA error handling mappings."""

    def test_error_types(self):
        from torchada._mapping import _MAPPING_RULE
        assert _MAPPING_RULE["cudaError_t"] == "musaError_t"
        assert _MAPPING_RULE["cudaSuccess"] == "musaSuccess"

    def test_error_functions(self):
        from torchada._mapping import _MAPPING_RULE
        assert _MAPPING_RULE["cudaGetLastError"] == "musaGetLastError"
        assert _MAPPING_RULE["cudaGetErrorString"] == "musaGetErrorString"
        assert _MAPPING_RULE["cudaPeekAtLastError"] == "musaPeekAtLastError"


class TestNCCLMappings:
    """Test NCCL to MCCL mappings."""

    def test_nccl_basic(self):
        from torchada._mapping import _MAPPING_RULE
        assert _MAPPING_RULE["nccl"] == "mccl"
        assert _MAPPING_RULE["NCCL"] == "MCCL"

    def test_nccl_types(self):
        from torchada._mapping import _MAPPING_RULE
        assert _MAPPING_RULE["ncclComm_t"] == "mcclComm_t"
        assert _MAPPING_RULE["ncclDataType_t"] == "mcclDataType_t"
        assert _MAPPING_RULE["ncclRedOp_t"] == "mcclRedOp_t"
        assert _MAPPING_RULE["ncclResult_t"] == "mcclResult_t"
        assert _MAPPING_RULE["ncclSuccess"] == "mcclSuccess"
        assert _MAPPING_RULE["ncclUniqueId"] == "mcclUniqueId"

    def test_nccl_comm_functions(self):
        from torchada._mapping import _MAPPING_RULE
        assert _MAPPING_RULE["ncclCommInitRank"] == "mcclCommInitRank"
        assert _MAPPING_RULE["ncclCommInitAll"] == "mcclCommInitAll"
        assert _MAPPING_RULE["ncclCommDestroy"] == "mcclCommDestroy"
        assert _MAPPING_RULE["ncclCommCount"] == "mcclCommCount"
        assert _MAPPING_RULE["ncclCommCuDevice"] == "mcclCommCuDevice"
        assert _MAPPING_RULE["ncclCommUserRank"] == "mcclCommUserRank"
        assert _MAPPING_RULE["ncclGetUniqueId"] == "mcclGetUniqueId"

    def test_nccl_collective_functions(self):
        from torchada._mapping import _MAPPING_RULE
        assert _MAPPING_RULE["ncclAllReduce"] == "mcclAllReduce"
        assert _MAPPING_RULE["ncclBroadcast"] == "mcclBroadcast"
        assert _MAPPING_RULE["ncclReduce"] == "mcclReduce"
        assert _MAPPING_RULE["ncclAllGather"] == "mcclAllGather"
        assert _MAPPING_RULE["ncclReduceScatter"] == "mcclReduceScatter"
        assert _MAPPING_RULE["ncclSend"] == "mcclSend"
        assert _MAPPING_RULE["ncclRecv"] == "mcclRecv"
        assert _MAPPING_RULE["ncclGroupStart"] == "mcclGroupStart"
        assert _MAPPING_RULE["ncclGroupEnd"] == "mcclGroupEnd"


class TestLibraryMappings:
    """Test other library mappings."""

    def test_cusparse(self):
        from torchada._mapping import _MAPPING_RULE
        assert _MAPPING_RULE["cusparse"] == "musparse"
        assert _MAPPING_RULE["CUSPARSE"] == "MUSPARSE"
        assert _MAPPING_RULE["cusparseHandle_t"] == "musparseHandle_t"
        assert _MAPPING_RULE["cusparseCreate"] == "musparseCreate"
        assert _MAPPING_RULE["cusparseDestroy"] == "musparseDestroy"

    def test_cusolver(self):
        from torchada._mapping import _MAPPING_RULE
        assert _MAPPING_RULE["cusolver"] == "musolver"
        assert _MAPPING_RULE["CUSOLVER"] == "MUSOLVER"
        assert _MAPPING_RULE["cusolverDnHandle_t"] == "musolverDnHandle_t"
        assert _MAPPING_RULE["cusolverDnCreate"] == "musolverDnCreate"
        assert _MAPPING_RULE["cusolverDnDestroy"] == "musolverDnDestroy"

    def test_cufft(self):
        from torchada._mapping import _MAPPING_RULE
        assert _MAPPING_RULE["cufft"] == "mufft"
        assert _MAPPING_RULE["CUFFT"] == "MUFFT"
        assert _MAPPING_RULE["cufftHandle"] == "mufftHandle"
        assert _MAPPING_RULE["cufftPlan1d"] == "mufftPlan1d"
        assert _MAPPING_RULE["cufftPlan2d"] == "mufftPlan2d"
        assert _MAPPING_RULE["cufftPlan3d"] == "mufftPlan3d"
        assert _MAPPING_RULE["cufftExecC2C"] == "mufftExecC2C"
        assert _MAPPING_RULE["cufftExecR2C"] == "mufftExecR2C"
        assert _MAPPING_RULE["cufftExecC2R"] == "mufftExecC2R"

    def test_cutlass(self):
        from torchada._mapping import _MAPPING_RULE
        assert _MAPPING_RULE["cutlass"] == "mutlass"
        assert _MAPPING_RULE["CUTLASS"] == "MUTLASS"
        assert _MAPPING_RULE["cutlass/"] == "mutlass/"
        assert _MAPPING_RULE["cutlass::"] == "mutlass::"

    def test_cub(self):
        from torchada._mapping import _MAPPING_RULE
        assert _MAPPING_RULE["cub::"] == "mub::"
        assert _MAPPING_RULE["cub/"] == "mub/"

    def test_thrust(self):
        from torchada._mapping import _MAPPING_RULE
        assert _MAPPING_RULE["thrust::cuda"] == "thrust::musa"


class TestIntrinsicMappings:
    """Test CUDA intrinsic and math function mappings."""

    def test_shuffle_intrinsics(self):
        from torchada._mapping import _MAPPING_RULE
        assert _MAPPING_RULE["__shfl_sync"] == "__shfl_sync"
        assert _MAPPING_RULE["__shfl_xor_sync"] == "__shfl_xor_sync"
        assert _MAPPING_RULE["__shfl_up_sync"] == "__shfl_up_sync"
        assert _MAPPING_RULE["__shfl_down_sync"] == "__shfl_down_sync"

    def test_vote_intrinsics(self):
        from torchada._mapping import _MAPPING_RULE
        assert _MAPPING_RULE["__ballot_sync"] == "__ballot_sync"
        assert _MAPPING_RULE["__any_sync"] == "__any_sync"
        assert _MAPPING_RULE["__all_sync"] == "__all_sync"

    def test_sync_intrinsics(self):
        from torchada._mapping import _MAPPING_RULE
        assert _MAPPING_RULE["__syncthreads"] == "__syncthreads"
        assert _MAPPING_RULE["__syncwarp"] == "__syncwarp"
        assert _MAPPING_RULE["__threadfence"] == "__threadfence"
        assert _MAPPING_RULE["__threadfence_block"] == "__threadfence_block"
        assert _MAPPING_RULE["__threadfence_system"] == "__threadfence_system"

    def test_atomic_operations(self):
        from torchada._mapping import _MAPPING_RULE
        assert _MAPPING_RULE["atomicAdd"] == "atomicAdd"
        assert _MAPPING_RULE["atomicSub"] == "atomicSub"
        assert _MAPPING_RULE["atomicExch"] == "atomicExch"
        assert _MAPPING_RULE["atomicMin"] == "atomicMin"
        assert _MAPPING_RULE["atomicMax"] == "atomicMax"
        assert _MAPPING_RULE["atomicInc"] == "atomicInc"
        assert _MAPPING_RULE["atomicDec"] == "atomicDec"
        assert _MAPPING_RULE["atomicCAS"] == "atomicCAS"
        assert _MAPPING_RULE["atomicAnd"] == "atomicAnd"
        assert _MAPPING_RULE["atomicOr"] == "atomicOr"
        assert _MAPPING_RULE["atomicXor"] == "atomicXor"

    def test_half_precision(self):
        from torchada._mapping import _MAPPING_RULE
        assert _MAPPING_RULE["__float2half"] == "__float2half"
        assert _MAPPING_RULE["__half2float"] == "__half2float"
        assert _MAPPING_RULE["__hadd"] == "__hadd"
        assert _MAPPING_RULE["__hsub"] == "__hsub"
        assert _MAPPING_RULE["__hmul"] == "__hmul"
        assert _MAPPING_RULE["__hdiv"] == "__hdiv"
        assert _MAPPING_RULE["__hfma"] == "__hfma"


class TestIncludeMappings:
    """Test header include mappings."""

    def test_cuda_headers(self):
        from torchada._mapping import _MAPPING_RULE
        assert _MAPPING_RULE["cuda_runtime.h"] == "musa_runtime.h"
        assert _MAPPING_RULE["cuda_runtime_api.h"] == "musa_runtime_api.h"
        assert _MAPPING_RULE["cuda.h"] == "musa.h"
        assert _MAPPING_RULE["cuda_fp16.h"] == "musa_fp16.h"
        assert _MAPPING_RULE["cuda_bf16.h"] == "musa_bf16.h"


class TestPyTorchCppMappings:
    """Test PyTorch C++ API mappings."""

    def test_pytorch_stream_utils(self):
        from torchada._mapping import _MAPPING_RULE
        assert _MAPPING_RULE["getCurrentCUDAStream"] == "getCurrentMUSAStream"
        assert _MAPPING_RULE["getDefaultCUDAStream"] == "getDefaultMUSAStream"
        assert _MAPPING_RULE["CUDAStream"] == "MUSAStream"

    def test_pytorch_guards(self):
        from torchada._mapping import _MAPPING_RULE
        assert _MAPPING_RULE["CUDAGuard"] == "MUSAGuard"
        assert _MAPPING_RULE["OptionalCUDAGuard"] == "OptionalMUSAGuard"
        assert _MAPPING_RULE["CUDAStreamGuard"] == "MUSAStreamGuard"
        assert _MAPPING_RULE["CUDAEvent"] == "MUSAEvent"

    def test_pytorch_torch_namespace(self):
        from torchada._mapping import _MAPPING_RULE
        assert _MAPPING_RULE["torch::cuda::getCurrentCUDAStream"] == "torch::musa::getCurrentMUSAStream"
        assert _MAPPING_RULE["torch::cuda::getDefaultCUDAStream"] == "torch::musa::getDefaultMUSAStream"
        assert _MAPPING_RULE["torch::cuda::getStreamFromPool"] == "torch::musa::getStreamFromPool"


class TestMappingCount:
    """Test total mapping count."""

    def test_mapping_count(self):
        from torchada._mapping import _MAPPING_RULE
        # We should have a substantial number of mappings
        assert len(_MAPPING_RULE) >= 250

    def test_ext_replaced_mapping(self):
        from torchada._mapping import EXT_REPLACED_MAPPING
        # Extensions are converted: .cu -> .mu, .cuh -> .muh for mcc compiler
        assert EXT_REPLACED_MAPPING["cu"] == "mu"
        assert EXT_REPLACED_MAPPING["cuh"] == "muh"
