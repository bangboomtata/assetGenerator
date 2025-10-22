from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os

print("=" * 80)
print("Building Intel XPU SYCL rasterizer (GPU only, with grid_neighbor)...")
print("=" * 80)

# Force Intel oneAPI compilers
os.environ["CXX"] = "icpx"
os.environ["CC"] = "icx"

# Source files for SYCL rasterizer
sources = [
    "lib/custom_rasterizer_kernel/dpct_output/rasterizer_gpu_full.dp.cpp",
    "lib/custom_rasterizer_kernel/grid_neighbor.cpp",
]

# Remove irrelevant GCC warnings & add SYCL/deprecation-related flags
flags_to_remove = [
    "-Wno-unused-but-set-variable",
    "-Wno-maybe-uninitialized",
]

extra_compile_args = {
    "cxx": [
        "-O2",
        "-std=c++17",
        "-fsycl",
        "-DUSE_SYCL",
        "-Wno-deprecated-declarations",   # suppress deprecated warnings
    ]
}

# Remove unwanted flags if they exist
for flag in flags_to_remove:
    if flag in extra_compile_args["cxx"]:
        extra_compile_args["cxx"].remove(flag)

extra_link_args = ["-fsycl"]

# Build setup
setup(
    name="custom_rasterizer_kernel",
    ext_modules=[
        CppExtension(
            name="custom_rasterizer_kernel",
            sources=sources,
            extra_compile_args=extra_compile_args["cxx"],
            extra_link_args=extra_link_args,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
