#ifndef RASTERIZER_H_
#define RASTERIZER_H_

#include <torch/extension.h>
#include <vector>

#define INT64 unsigned long long
#define MAXINT 2147483647

inline float calculateSignedArea2(float* a, float* b, float* c) {
    return ((c[0] - a[0]) * (b[1] - a[1]) - (b[0] - a[0]) * (c[1] - a[1]));
}

inline void calculateBarycentricCoordinate(float* a, float* b, float* c, float* p, float* barycentric) {
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

inline bool isBarycentricCoordInBounds(float* barycentricCoord) {
    return barycentricCoord[0] >= 0.0 && barycentricCoord[0] <= 1.0 &&
           barycentricCoord[1] >= 0.0 && barycentricCoord[1] <= 1.0 &&
           barycentricCoord[2] >= 0.0 && barycentricCoord[2] <= 1.0;
}

// Only the functions you actually need
std::vector<torch::Tensor> rasterize_image(torch::Tensor V, torch::Tensor F, torch::Tensor D,
    int width, int height, float occlusion_truncation, int use_depth_prior);

torch::Tensor rasterize(torch::Tensor vertices, torch::Tensor faces, torch::Tensor colors, 
                       int height, int width, float near, int far);

#endif