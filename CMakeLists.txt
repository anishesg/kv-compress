cmake_minimum_required(VERSION 3.18)
project(kv_compress CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Require sm_80 (Ampere) minimum; also support sm_86, sm_89, sm_90
set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90)

# Release flags
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Useful diagnostics during development
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} \
    --expt-relaxed-constexpr \
    --expt-extended-lambda \
    -Xcompiler -Wall \
    -lineinfo")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

include_directories(src)

# Shared library of all kernels
add_library(kv_compress_kernels STATIC
    src/codebook_train.cu
    src/quantize.cu
    src/attention_ref.cu
    src/attention_mixed.cu
)

set_target_properties(kv_compress_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Correctness test
add_executable(test_correctness tests/test_correctness.cu)
target_link_libraries(test_correctness PRIVATE kv_compress_kernels)
set_target_properties(test_correctness PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Memory footprint analysis
add_executable(memory_analysis benchmarks/memory_analysis.cu)
target_link_libraries(memory_analysis PRIVATE kv_compress_kernels)
set_target_properties(memory_analysis PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
