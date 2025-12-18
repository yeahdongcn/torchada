"""
CUDA to MUSA mapping rules for source code porting.

This module contains the comprehensive mapping dictionary for converting
CUDA-specific symbols to their MUSA equivalents during extension builds.
"""

# Extension file suffix mappings (convert .cu/.cuh to .mu/.muh for MUSA compiler)
# The mcc compiler requires .mu/.muh extensions to properly compile MUSA code
EXT_REPLACED_MAPPING = {
    "cuh": "muh",
    "cu": "mu",
}

# Comprehensive CUDA to MUSA symbol mapping
_MAPPING_RULE = {
    # ATen CUDA -> MUSA (stream functions are in c10::musa)
    "at::cuda::getCurrentCUDAStream": "c10::musa::getCurrentMUSAStream",
    "at::cuda::getDefaultCUDAStream": "c10::musa::getDefaultMUSAStream",
    "at::cuda::setCurrentCUDAStream": "c10::musa::setCurrentMUSAStream",
    "at::cuda": "at::musa",
    "at::cuda::": "at::musa::",
    "ATen/cuda": "ATen/musa",
    "ATen/CUDAContext.h": "ATen/musa/MUSAContext.h",
    "ATen/CUDAGeneratorImpl.h": "ATen/musa/MUSAGeneratorImpl.h",

    # C10 CUDA -> MUSA
    "c10::cuda::getCurrentCUDAStream": "c10::musa::getCurrentMUSAStream",
    "c10::cuda::getDefaultCUDAStream": "c10::musa::getDefaultMUSAStream",
    "c10::cuda::setCurrentCUDAStream": "c10::musa::setCurrentMUSAStream",
    "c10::cuda": "c10::musa",
    "c10::cuda::": "c10::musa::",
    "c10/cuda": "c10/musa",
    "<c10/cuda/CUDAStream.h>": '"torch_musa/csrc/core/MUSAStream.h"',

    # CUDA namespaces and paths
    "torch::cuda": "torch::musa",
    "torch.cuda": "torch.musa",
    "at::kCUDA": "at::kMUSA",
    "at::DeviceType::CUDA": "at::DeviceType::MUSA",
    "c10::DeviceType::CUDA": "c10::DeviceType::MUSA",

    # cuBLAS -> muBLAS
    "cublas": "mublas",
    "CUBLAS": "MUBLAS",
    "cublasHandle_t": "mublasHandle_t",
    "cublasStatus_t": "mublasStatus_t",
    "cublasCreate": "mublasCreate",
    "cublasDestroy": "mublasDestroy",
    "cublasSetStream": "mublasSetStream",
    "cublasGetStream": "mublasGetStream",
    "cublasSgemm": "mublasSgemm",
    "cublasDgemm": "mublasDgemm",
    "cublasHgemm": "mublasHgemm",
    "cublasGemmEx": "mublasGemmEx",
    "cublasGemmBatchedEx": "mublasGemmBatchedEx",
    "cublasGemmStridedBatchedEx": "mublasGemmStridedBatchedEx",
    "cublasSgemmBatched": "mublasSgemmBatched",
    "cublasDgemmBatched": "mublasDgemmBatched",
    "cublasSgemmStridedBatched": "mublasSgemmStridedBatched",
    "cublasDgemmStridedBatched": "mublasDgemmStridedBatched",

    # cuRAND -> muRAND
    "curand": "murand",
    "CURAND": "MURAND",
    "curandState": "murandState",
    "curandStatePhilox4_32_10_t": "murandStatePhilox4_32_10_t",
    "curand_init": "murand_init",
    "curand_uniform": "murand_uniform",
    "curand_uniform4": "murand_uniform4",
    "curand_normal": "murand_normal",
    "curand_normal4": "murand_normal4",

    # cuDNN -> muDNN
    "cudnn": "mudnn",
    "CUDNN": "MUDNN",
    "cudnnHandle_t": "mudnnHandle_t",
    "cudnnCreate": "mudnnCreate",
    "cudnnDestroy": "mudnnDestroy",

    # CUDA Runtime
    "cudaMalloc": "musaMalloc",
    "cudaFree": "musaFree",
    "cudaMemcpy": "musaMemcpy",
    "cudaMemcpyAsync": "musaMemcpyAsync",
    "cudaMemset": "musaMemset",
    "cudaMemsetAsync": "musaMemsetAsync",
    "cudaDeviceSynchronize": "musaDeviceSynchronize",
    "cudaStreamSynchronize": "musaStreamSynchronize",
    "cudaGetDevice": "musaGetDevice",
    "cudaSetDevice": "musaSetDevice",
    "cudaGetDeviceCount": "musaGetDeviceCount",
    "cudaGetDeviceProperties": "musaGetDeviceProperties",
    "cudaDeviceGetAttribute": "musaDeviceGetAttribute",

    # CUDA Stream/Event
    "cudaStream_t": "musaStream_t",
    "cudaEvent_t": "musaEvent_t",
    "cudaStreamCreate": "musaStreamCreate",
    "cudaStreamDestroy": "musaStreamDestroy",
    "cudaEventCreate": "musaEventCreate",
    "cudaEventDestroy": "musaEventDestroy",
    "cudaEventRecord": "musaEventRecord",
    "cudaEventSynchronize": "musaEventSynchronize",
    "cudaEventElapsedTime": "musaEventElapsedTime",
    "cudaStreamWaitEvent": "musaStreamWaitEvent",

    # CUDA Error handling
    "cudaError_t": "musaError_t",
    "cudaSuccess": "musaSuccess",
    "cudaGetLastError": "musaGetLastError",
    "cudaGetErrorString": "musaGetErrorString",
    "cudaPeekAtLastError": "musaPeekAtLastError",

    # CUDA Memory types
    "cudaMemcpyHostToDevice": "musaMemcpyHostToDevice",
    "cudaMemcpyDeviceToHost": "musaMemcpyDeviceToHost",
    "cudaMemcpyDeviceToDevice": "musaMemcpyDeviceToDevice",
    "cudaMemcpyHostToHost": "musaMemcpyHostToHost",

    # Data types
    "__half": "__half",  # Keep as-is, MUSA supports __half
    "half": "half",
    "__nv_bfloat16": "__mt_bfloat16",
    "nv_bfloat16": "mt_bfloat16",

    # Cutlass -> Mutlass
    "cutlass": "mutlass",
    "CUTLASS": "MUTLASS",
    "cutlass/": "mutlass/",
    "cutlass::": "mutlass::",

    # CUB -> MUB (CUDA Unbound)
    "cub::": "mub::",
    "cub/": "mub/",

    # Thrust
    "thrust::cuda": "thrust::musa",

    # NCCL -> MCCL
    "nccl": "mccl",
    "NCCL": "MCCL",
    "ncclComm_t": "mcclComm_t",
    "ncclDataType_t": "mcclDataType_t",
    "ncclRedOp_t": "mcclRedOp_t",
    "ncclResult_t": "mcclResult_t",
    "ncclSuccess": "mcclSuccess",

    # cuSPARSE -> muSPARSE
    "cusparse": "musparse",
    "CUSPARSE": "MUSPARSE",
    "cusparseHandle_t": "musparseHandle_t",
    "cusparseCreate": "musparseCreate",
    "cusparseDestroy": "musparseDestroy",

    # cuSOLVER -> muSOLVER
    "cusolver": "musolver",
    "CUSOLVER": "MUSOLVER",
    "cusolverDnHandle_t": "musolverDnHandle_t",
    "cusolverDnCreate": "musolverDnCreate",
    "cusolverDnDestroy": "musolverDnDestroy",

    # cuFFT -> muFFT
    "cufft": "mufft",
    "CUFFT": "MUFFT",
    "cufftHandle": "mufftHandle",
    "cufftPlan1d": "mufftPlan1d",
    "cufftPlan2d": "mufftPlan2d",
    "cufftPlan3d": "mufftPlan3d",
    "cufftExecC2C": "mufftExecC2C",
    "cufftExecR2C": "mufftExecR2C",
    "cufftExecC2R": "mufftExecC2R",

    # CUDA kernel launch syntax
    "<<<": "<<<",  # Keep as-is, MUSA uses same syntax
    ">>>": ">>>",

    # CUDA device attributes
    "cudaDevAttrMaxThreadsPerBlock": "musaDevAttrMaxThreadsPerBlock",
    "cudaDevAttrMaxBlockDimX": "musaDevAttrMaxBlockDimX",
    "cudaDevAttrMaxBlockDimY": "musaDevAttrMaxBlockDimY",
    "cudaDevAttrMaxBlockDimZ": "musaDevAttrMaxBlockDimZ",
    "cudaDevAttrMaxGridDimX": "musaDevAttrMaxGridDimX",
    "cudaDevAttrMaxGridDimY": "musaDevAttrMaxGridDimY",
    "cudaDevAttrMaxGridDimZ": "musaDevAttrMaxGridDimZ",
    "cudaDevAttrMaxSharedMemoryPerBlock": "musaDevAttrMaxSharedMemoryPerBlock",
    "cudaDevAttrWarpSize": "musaDevAttrWarpSize",
    "cudaDevAttrMultiProcessorCount": "musaDevAttrMultiProcessorCount",

    # Additional PyTorch CUDA utilities
    "getCurrentCUDAStream": "getCurrentMUSAStream",
    "getDefaultCUDAStream": "getDefaultMUSAStream",
    "CUDAStream": "MUSAStream",
    "CUDAGuard": "MUSAGuard",
    "OptionalCUDAGuard": "OptionalMUSAGuard",
    "CUDAStreamGuard": "MUSAStreamGuard",
    "CUDAEvent": "MUSAEvent",

    # CUDA includes to MUSA
    "cuda_runtime.h": "musa_runtime.h",
    "cuda_runtime_api.h": "musa_runtime_api.h",
    "cuda.h": "musa.h",
    "cuda_fp16.h": "musa_fp16.h",
    "cuda_bf16.h": "musa_bf16.h",

    # Additional CUDA runtime functions
    "cudaHostAlloc": "musaHostAlloc",
    "cudaHostFree": "musaHostFree",
    "cudaMallocHost": "musaMallocHost",
    "cudaFreeHost": "musaFreeHost",
    "cudaMallocManaged": "musaMallocManaged",
    "cudaMallocAsync": "musaMallocAsync",
    "cudaFreeAsync": "musaFreeAsync",
    "cudaMemcpy2D": "musaMemcpy2D",
    "cudaMemcpy2DAsync": "musaMemcpy2DAsync",
    "cudaMemcpy3D": "musaMemcpy3D",
    "cudaMemcpy3DAsync": "musaMemcpy3DAsync",
    "cudaMemGetInfo": "musaMemGetInfo",
    "cudaMemPrefetchAsync": "musaMemPrefetchAsync",
    "cudaPointerGetAttributes": "musaPointerGetAttributes",

    # CUDA stream flags and types
    "cudaStreamDefault": "musaStreamDefault",
    "cudaStreamNonBlocking": "musaStreamNonBlocking",
    "cudaStreamCreateWithFlags": "musaStreamCreateWithFlags",
    "cudaStreamCreateWithPriority": "musaStreamCreateWithPriority",
    "cudaStreamQuery": "musaStreamQuery",
    "cudaStreamGetPriority": "musaStreamGetPriority",
    "cudaStreamGetFlags": "musaStreamGetFlags",

    # CUDA event flags
    "cudaEventDefault": "musaEventDefault",
    "cudaEventBlockingSync": "musaEventBlockingSync",
    "cudaEventDisableTiming": "musaEventDisableTiming",
    "cudaEventCreateWithFlags": "musaEventCreateWithFlags",
    "cudaEventQuery": "musaEventQuery",

    # CUDA memory flags
    "cudaHostAllocDefault": "musaHostAllocDefault",
    "cudaHostAllocPortable": "musaHostAllocPortable",
    "cudaHostAllocMapped": "musaHostAllocMapped",
    "cudaHostAllocWriteCombined": "musaHostAllocWriteCombined",
    "cudaMemoryTypeHost": "musaMemoryTypeHost",
    "cudaMemoryTypeDevice": "musaMemoryTypeDevice",
    "cudaMemoryTypeManaged": "musaMemoryTypeManaged",

    # Device management
    "cudaDeviceReset": "musaDeviceReset",
    "cudaDeviceSetCacheConfig": "musaDeviceSetCacheConfig",
    "cudaDeviceGetCacheConfig": "musaDeviceGetCacheConfig",
    "cudaDeviceSetSharedMemConfig": "musaDeviceSetSharedMemConfig",
    "cudaDeviceGetSharedMemConfig": "musaDeviceGetSharedMemConfig",
    "cudaGetDeviceFlags": "musaGetDeviceFlags",
    "cudaSetDeviceFlags": "musaSetDeviceFlags",
    "cudaDeviceCanAccessPeer": "musaDeviceCanAccessPeer",
    "cudaDeviceEnablePeerAccess": "musaDeviceEnablePeerAccess",
    "cudaDeviceDisablePeerAccess": "musaDeviceDisablePeerAccess",

    # CUDA occupancy
    "cudaOccupancyMaxActiveBlocksPerMultiprocessor": "musaOccupancyMaxActiveBlocksPerMultiprocessor",
    "cudaOccupancyMaxPotentialBlockSize": "musaOccupancyMaxPotentialBlockSize",

    # Additional device properties
    "cudaDeviceProp": "musaDeviceProp",
    "cudaFuncAttributes": "musaFuncAttributes",
    "cudaFuncGetAttributes": "musaFuncGetAttributes",
    "cudaFuncSetAttribute": "musaFuncSetAttribute",
    "cudaFuncSetCacheConfig": "musaFuncSetCacheConfig",

    # CUDA texture/surface (if supported)
    "cudaTextureObject_t": "musaTextureObject_t",
    "cudaSurfaceObject_t": "musaSurfaceObject_t",
    "cudaCreateTextureObject": "musaCreateTextureObject",
    "cudaDestroyTextureObject": "musaDestroyTextureObject",
    "cudaCreateSurfaceObject": "musaCreateSurfaceObject",
    "cudaDestroySurfaceObject": "musaDestroySurfaceObject",

    # CUDA cooperative groups (if supported)
    "cooperative_groups": "cooperative_groups",

    # Additional cuBLAS functions
    "cublasSetMathMode": "mublasSetMathMode",
    "cublasGetMathMode": "mublasGetMathMode",
    "CUBLAS_DEFAULT_MATH": "MUBLAS_DEFAULT_MATH",
    "CUBLAS_TENSOR_OP_MATH": "MUBLAS_TENSOR_OP_MATH",
    "cublasLtCreate": "mublasLtCreate",
    "cublasLtDestroy": "mublasLtDestroy",
    "cublasLtHandle_t": "mublasLtHandle_t",
    "cublasLtMatmul": "mublasLtMatmul",

    # Additional cuDNN functions
    "cudnnStatus_t": "mudnnStatus_t",
    "cudnnSetStream": "mudnnSetStream",
    "cudnnGetStream": "mudnnGetStream",
    "cudnnTensorDescriptor_t": "mudnnTensorDescriptor_t",
    "cudnnFilterDescriptor_t": "mudnnFilterDescriptor_t",
    "cudnnConvolutionDescriptor_t": "mudnnConvolutionDescriptor_t",
    "cudnnPoolingDescriptor_t": "mudnnPoolingDescriptor_t",
    "cudnnActivationDescriptor_t": "mudnnActivationDescriptor_t",
    "cudnnDropoutDescriptor_t": "mudnnDropoutDescriptor_t",
    "cudnnRNNDescriptor_t": "mudnnRNNDescriptor_t",
    "cudnnCreateTensorDescriptor": "mudnnCreateTensorDescriptor",
    "cudnnDestroyTensorDescriptor": "mudnnDestroyTensorDescriptor",
    "cudnnSetTensor4dDescriptor": "mudnnSetTensor4dDescriptor",
    "cudnnSetTensorNdDescriptor": "mudnnSetTensorNdDescriptor",

    # Flash attention and transformer related (common in vLLM/SGLang)
    "flash_attn": "flash_attn",  # Usually kept as-is
    "FlashAttnFunc": "FlashAttnFunc",
    "FlashAttnQKVPackedFunc": "FlashAttnQKVPackedFunc",

    # Paged attention (vLLM specific)
    "paged_attention_v1": "paged_attention_v1",
    "paged_attention_v2": "paged_attention_v2",

    # Additional NCCL functions
    "ncclCommInitRank": "mcclCommInitRank",
    "ncclCommInitAll": "mcclCommInitAll",
    "ncclCommDestroy": "mcclCommDestroy",
    "ncclCommCount": "mcclCommCount",
    "ncclCommCuDevice": "mcclCommCuDevice",
    "ncclCommUserRank": "mcclCommUserRank",
    "ncclAllReduce": "mcclAllReduce",
    "ncclBroadcast": "mcclBroadcast",
    "ncclReduce": "mcclReduce",
    "ncclAllGather": "mcclAllGather",
    "ncclReduceScatter": "mcclReduceScatter",
    "ncclGroupStart": "mcclGroupStart",
    "ncclGroupEnd": "mcclGroupEnd",
    "ncclSend": "mcclSend",
    "ncclRecv": "mcclRecv",
    "ncclGetUniqueId": "mcclGetUniqueId",
    "ncclUniqueId": "mcclUniqueId",

    # CUDA math intrinsics
    "__shfl_sync": "__shfl_sync",  # Usually same syntax
    "__shfl_xor_sync": "__shfl_xor_sync",
    "__shfl_up_sync": "__shfl_up_sync",
    "__shfl_down_sync": "__shfl_down_sync",
    "__ballot_sync": "__ballot_sync",
    "__any_sync": "__any_sync",
    "__all_sync": "__all_sync",
    "__syncthreads": "__syncthreads",
    "__syncwarp": "__syncwarp",
    "__threadfence": "__threadfence",
    "__threadfence_block": "__threadfence_block",
    "__threadfence_system": "__threadfence_system",

    # Atomic operations
    "atomicAdd": "atomicAdd",
    "atomicSub": "atomicSub",
    "atomicExch": "atomicExch",
    "atomicMin": "atomicMin",
    "atomicMax": "atomicMax",
    "atomicInc": "atomicInc",
    "atomicDec": "atomicDec",
    "atomicCAS": "atomicCAS",
    "atomicAnd": "atomicAnd",
    "atomicOr": "atomicOr",
    "atomicXor": "atomicXor",

    # CUDA math functions
    "__float2half": "__float2half",
    "__half2float": "__half2float",
    "__float2half_rn": "__float2half_rn",
    "__float22half2_rn": "__float22half2_rn",
    "__half22float2": "__half22float2",
    "__hadd": "__hadd",
    "__hsub": "__hsub",
    "__hmul": "__hmul",
    "__hdiv": "__hdiv",
    "__hfma": "__hfma",
    "__hadd2": "__hadd2",
    "__hsub2": "__hsub2",
    "__hmul2": "__hmul2",
    "__hfma2": "__hfma2",

    # Common macros
    "CUDA_KERNEL_LOOP": "MUSA_KERNEL_LOOP",
    "CUDA_1D_KERNEL_LOOP": "MUSA_1D_KERNEL_LOOP",
    "CUDA_2D_KERNEL_LOOP": "MUSA_2D_KERNEL_LOOP",
    "CUDA_NUM_THREADS": "MUSA_NUM_THREADS",
    "GET_BLOCKS": "GET_BLOCKS",
    "DIVUP": "DIVUP",

    # PyTorch C++ API cuda utilities
    "torch::cuda::getCurrentCUDAStream": "torch::musa::getCurrentMUSAStream",
    "torch::cuda::getDefaultCUDAStream": "torch::musa::getDefaultMUSAStream",
    "torch::cuda::getStreamFromPool": "torch::musa::getStreamFromPool",
    "torch::kCUDA": "torch::kMUSA",

    # Device index utilities
    "cudaDeviceIndex": "musaDeviceIndex",
    "CUDADeviceIndex": "MUSADeviceIndex",
}

