#include <torch/extension.h>
#include "rasterizer.h"

// Declare the function from your rasterizer source
std::vector<torch::Tensor> rasterize_image_gpu(
    torch::Tensor V,
    torch::Tensor F,
    torch::Tensor D,
    int width,
    int height,
    float occlusion_truncation,
    int use_depth_prior
);

// Pybind11 wrapper
PYBIND11_MODULE(custom_rasterizer_kernel, m) {
    m.def("rasterize", &rasterize_image_gpu, "Rasterize Image GPU");
}
