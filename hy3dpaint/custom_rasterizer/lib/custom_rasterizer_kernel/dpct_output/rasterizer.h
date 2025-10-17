#ifndef RASTERIZER_H_
#define RASTERIZER_H_

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <torch/extension.h>
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <vector>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#define INT64 unsigned long long
#define MAXINT 2147483647

#ifdef SYCL_LANGUAGE_VERSION
#define HOST_DEVICE 
#else
#define HOST_DEVICE
#endif

HOST_DEVICE inline float calculateSignedArea2(float* a, float* b, float* c) {
    return ((c[0] - a[0]) * (b[1] - a[1]) - (b[0] - a[0]) * (c[1] - a[1]));
}

HOST_DEVICE inline void calculateBarycentricCoordinate(float* a, float* b, float* c, float* p,
    float* barycentric)
{
    float beta_tri = calculateSignedArea2(a, p, c);
    float gamma_tri = calculateSignedArea2(a, b, p);
    float area = calculateSignedArea2(a, b, c);
    if (area == 0) {
        barycentric[0] = barycentric[1] = barycentric[2] = -1.0;
        return;
    }
    float tri_inv = 1.0 / area;
    barycentric[0] = 1.0 - (beta_tri + gamma_tri) * tri_inv;
    barycentric[1] = beta_tri * tri_inv;
    barycentric[2] = gamma_tri * tri_inv;
}

HOST_DEVICE inline bool isBarycentricCoordInBounds(float* barycentricCoord) {
    return barycentricCoord[0] >= 0.0 && barycentricCoord[0] <= 1.0 &&
           barycentricCoord[1] >= 0.0 && barycentricCoord[1] <= 1.0 &&
           barycentricCoord[2] >= 0.0 && barycentricCoord[2] <= 1.0;
}

// Forward declarations
std::vector<torch::Tensor> rasterize_image_gpu(torch::Tensor V, torch::Tensor F, torch::Tensor D,
    int width, int height, float occlusion_truncation, int use_depth_prior);

std::vector<std::vector<torch::Tensor>> build_hierarchy(std::vector<torch::Tensor> view_layer_positions, std::vector<torch::Tensor> view_layer_normals, int num_level, int resolution);

std::vector<std::vector<torch::Tensor>> build_hierarchy_with_feat(
    std::vector<torch::Tensor> view_layer_positions,
    std::vector<torch::Tensor> view_layer_normals,
    std::vector<torch::Tensor> view_layer_feats,
    int num_level, int resolution);

#endif
