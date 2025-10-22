from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os

print("=" * 80)
print("Building Intel XPU SYCL rasterizer (GPU only, with grid_neighbor)...")
print("=" * 80)

# Force Intel oneAPI compilers
os.environ["CXX"] = "icpx"
os.environ["CC"] = "icx"

# Source files
sources = [
    "lib/custom_rasterizer_kernel/dpct_output/rasterizer_gpu_full.dp.cpp",
    "lib/custom_rasterizer_kernel/grid_neighbor.cpp",
    "lib/custom_rasterizer_kernel/dpct_output/pybind_wrapper.cpp",
]

# Compiler flags
extra_compile_args = [
    "-O2",
    "-std=c++17",
    "-fsycl",
    "-DUSE_SYCL",
    "-Wno-deprecated-declarations",
]

# Linker flags
extra_link_args = ["-fsycl"]

setup(
    name="custom_rasterizer_kernel",
    ext_modules=[
        CppExtension(
            name="custom_rasterizer_kernel",
            sources=sources,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
