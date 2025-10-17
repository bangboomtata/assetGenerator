#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "rasterizer.h"
#include <cmath>

void rasterizeTriangleGPU(int idx, float* vt0, float* vt1, float* vt2, int width, int height, INT64* zbuffer, float* d, float occlusion_truncation) {
    float x_min = std::min(vt0[0], std::min(vt1[0],vt2[0]));
    float x_max = std::max(vt0[0], std::max(vt1[0],vt2[0]));
    float y_min = std::min(vt0[1], std::min(vt1[1],vt2[1]));
    float y_max = std::max(vt0[1], std::max(vt1[1],vt2[1]));

    for (int px = x_min; px < x_max + 1; ++px) {
        if (px < 0 || px >= width)
            continue;
        for (int py = y_min; py < y_max + 1; ++py) {
            if (py < 0 || py >= height)
                continue;
            float vt[2] = {px + 0.5f, py + 0.5f};
            float baryCentricCoordinate[3];
            calculateBarycentricCoordinate(vt0, vt1, vt2, vt, baryCentricCoordinate);
            if (isBarycentricCoordInBounds(baryCentricCoordinate)) {
                int pixel = py * width + px;
                if (zbuffer == 0) {
                    dpct::atomic_exchange<
                        sycl::access::address_space::generic_space>(
                        &zbuffer[pixel], (INT64)(idx + 1));
                    continue;
                }
                float depth = baryCentricCoordinate[0] * vt0[2] + baryCentricCoordinate[1] * vt1[2] + baryCentricCoordinate[2] * vt2[2];
                float depth_thres = 0;
                if (d) {
                    depth_thres = d[pixel] * 0.49999f + 0.5f + occlusion_truncation;
                }
                
                int z_quantize = depth * (2<<17);
                INT64 token = (INT64)z_quantize * MAXINT + (INT64)(idx + 1);
                if (depth < depth_thres)
                    continue;
                dpct::atomic_fetch_min<
                    sycl::access::address_space::generic_space>(&zbuffer[pixel],
                                                                token);
            }
        }
    }
}

void barycentricFromImgcoordGPU(float* V, int* F, int* findices, INT64* zbuffer, int width, int height, int num_vertices, int num_faces,
    float* barycentric_map)
{
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int pix = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
              item_ct1.get_local_id(2);
    if (pix >= width * height)
        return;
    INT64 f = zbuffer[pix] % MAXINT;
    if (f == (MAXINT-1)) {
        findices[pix] = 0;
        barycentric_map[pix * 3] = 0;
        barycentric_map[pix * 3 + 1] = 0;
        barycentric_map[pix * 3 + 2] = 0;
        return;
    }
    findices[pix] = f;
    f -= 1;
    float barycentric[3] = {0, 0, 0};
    if (f >= 0) {
        float vt[2] = {float(pix % width) + 0.5f, float(pix / width) + 0.5f};
        float* vt0_ptr = V + (F[f * 3] * 4);
        float* vt1_ptr = V + (F[f * 3 + 1] * 4);
        float* vt2_ptr = V + (F[f * 3 + 2] * 4);

        float vt0[2] = {(vt0_ptr[0] / vt0_ptr[3] * 0.5f + 0.5f) * (width - 1) + 0.5f, (0.5f + 0.5f * vt0_ptr[1] / vt0_ptr[3]) * (height - 1) + 0.5f};
        float vt1[2] = {(vt1_ptr[0] / vt1_ptr[3] * 0.5f + 0.5f) * (width - 1) + 0.5f, (0.5f + 0.5f * vt1_ptr[1] / vt1_ptr[3]) * (height - 1) + 0.5f};
        float vt2[2] = {(vt2_ptr[0] / vt2_ptr[3] * 0.5f + 0.5f) * (width - 1) + 0.5f, (0.5f + 0.5f * vt2_ptr[1] / vt2_ptr[3]) * (height - 1) + 0.5f};

        calculateBarycentricCoordinate(vt0, vt1, vt2, vt, barycentric);

        barycentric[0] = barycentric[0] / vt0_ptr[3];
        barycentric[1] = barycentric[1] / vt1_ptr[3];
        barycentric[2] = barycentric[2] / vt2_ptr[3];
        float w = 1.0f / (barycentric[0] + barycentric[1] + barycentric[2]);
        barycentric[0] *= w;
        barycentric[1] *= w;
        barycentric[2] *= w;

    }
    barycentric_map[pix * 3] = barycentric[0];
    barycentric_map[pix * 3 + 1] = barycentric[1];
    barycentric_map[pix * 3 + 2] = barycentric[2];
}

void rasterizeImagecoordsKernelGPU(float* V, int* F, float* d, INT64* zbuffer, float occlusion_trunc, int width, int height, int num_vertices, int num_faces)
{
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int f = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    if (f >= num_faces)
        return; 

    float* vt0_ptr = V + (F[f * 3] * 4);
    float* vt1_ptr = V + (F[f * 3 + 1] * 4);
    float* vt2_ptr = V + (F[f * 3 + 2] * 4);

    float vt0[3] = {(vt0_ptr[0] / vt0_ptr[3] * 0.5f + 0.5f) * (width - 1) + 0.5f, (0.5f + 0.5f * vt0_ptr[1] / vt0_ptr[3]) * (height - 1) + 0.5f, vt0_ptr[2] / vt0_ptr[3] * 0.49999f + 0.5f};
    float vt1[3] = {(vt1_ptr[0] / vt1_ptr[3] * 0.5f + 0.5f) * (width - 1) + 0.5f, (0.5f + 0.5f * vt1_ptr[1] / vt1_ptr[3]) * (height - 1) + 0.5f, vt1_ptr[2] / vt1_ptr[3] * 0.49999f + 0.5f};
    float vt2[3] = {(vt2_ptr[0] / vt2_ptr[3] * 0.5f + 0.5f) * (width - 1) + 0.5f, (0.5f + 0.5f * vt2_ptr[1] / vt2_ptr[3]) * (height - 1) + 0.5f, vt2_ptr[2] / vt2_ptr[3] * 0.49999f + 0.5f};

    rasterizeTriangleGPU(f, vt0, vt1, vt2, width, height, zbuffer, d, occlusion_trunc);
}

std::vector<torch::Tensor> rasterize_image_gpu(torch::Tensor V, torch::Tensor F, torch::Tensor D,
    int width, int height, float occlusion_truncation, int use_depth_prior)
{
    int device_id = V.get_device();
    /*
    DPCT1093:0: The "device_id" device may be not the one intended for use.
    Adjust the selected device if needed.
    */
    dpct::select_device(device_id);
    int num_faces = F.size(0);
    int num_vertices = V.size(0);
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, device_id).requires_grad(false);
    auto INT64_options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA, device_id).requires_grad(false);
    auto findices = torch::zeros({height, width}, options);
    INT64 maxint = (INT64)MAXINT * (INT64)MAXINT + (MAXINT - 1);
    auto z_min = torch::ones({height, width}, INT64_options) * (long)maxint;

    if (!use_depth_prior) {
        ((sycl::queue *)(at::cuda::getCurrentCUDAStream()))
            ->submit([&](sycl::handler &cgh) {
                // helper variables defined
                auto z_min_data_ptr_long_ct3 = (INT64 *)z_min.data_ptr<long>();

                cgh.parallel_for(
                    sycl::nd_range<3>(
                        sycl::range<3>(1, 1, (num_faces + 255) / 256) *
                            sycl::range<3>(1, 1, 256),
                        sycl::range<3>(1, 1, 256)),
                    [=](sycl::nd_item<3> item_ct1) {
                        rasterizeImagecoordsKernelGPU(
                            V.data_ptr<float>(), F.data_ptr<int>(), 0,
                            z_min_data_ptr_long_ct3, occlusion_truncation,
                            width, height, num_vertices, num_faces);
                    });
            });
    } else {
        ((sycl::queue *)(at::cuda::getCurrentCUDAStream()))
            ->submit([&](sycl::handler &cgh) {
                // helper variables defined
                auto z_min_data_ptr_long_ct3 = (INT64 *)z_min.data_ptr<long>();

                cgh.parallel_for(
                    sycl::nd_range<3>(
                        sycl::range<3>(1, 1, (num_faces + 255) / 256) *
                            sycl::range<3>(1, 1, 256),
                        sycl::range<3>(1, 1, 256)),
                    [=](sycl::nd_item<3> item_ct1) {
                        rasterizeImagecoordsKernelGPU(
                            V.data_ptr<float>(), F.data_ptr<int>(),
                            D.data_ptr<float>(), z_min_data_ptr_long_ct3,
                            occlusion_truncation, width, height, num_vertices,
                            num_faces);
                    });
            });
    }

    auto float_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, device_id).requires_grad(false);
    auto barycentric = torch::zeros({height, width, 3}, float_options);
    dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
        // helper variables defined
        auto findices_data_ptr_int_ct2 = findices.data_ptr<int>();
        auto z_min_data_ptr_long_ct3 = (INT64 *)z_min.data_ptr<long>();
        auto barycentric_data_ptr_float_ct8 = barycentric.data_ptr<float>();

        cgh.parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, (width * height + 255) / 256) *
                    sycl::range<3>(1, 1, 256),
                sycl::range<3>(1, 1, 256)),
            [=](sycl::nd_item<3> item_ct1) {
                barycentricFromImgcoordGPU(
                    V.data_ptr<float>(), F.data_ptr<int>(),
                    findices_data_ptr_int_ct2, z_min_data_ptr_long_ct3, width,
                    height, num_vertices, num_faces,
                    barycentric_data_ptr_float_ct8);
            });
    });

    return {findices, barycentric};
}
