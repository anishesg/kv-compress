import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

src_dir = os.path.join(os.path.dirname(__file__), "src")
csrc_dir = os.path.join(os.path.dirname(__file__), "csrc")

nvcc_flags = [
    "-O3",
    "--use_fast_math",
    "-gencode=arch=compute_80,code=sm_80",
    "-gencode=arch=compute_86,code=sm_86",
    "-gencode=arch=compute_89,code=sm_89",
    "-gencode=arch=compute_90,code=sm_90",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
]

ext_modules = [
    CUDAExtension(
        name="kv_compress._C",
        sources=[
            os.path.join(csrc_dir, "bindings.cpp"),
            os.path.join(src_dir, "codebook_train.cu"),
            os.path.join(src_dir, "quantize.cu"),
            os.path.join(src_dir, "attention_mixed.cu"),
            os.path.join(src_dir, "attention_ref.cu"),
        ],
        include_dirs=[src_dir],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "nvcc": nvcc_flags,
        },
        libraries=["curand"],
    )
]

setup(
    name="kv_compress",
    version="0.1.0",
    description="Online KV-cache compression via per-head learned codebooks",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.8",
    install_requires=["torch>=2.0"],
)
