#include "rayreflection_render.h"
#include <c10/core/ScalarType.h>
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/native/quantized/cpu/embedding_packed_params.h>
#include <ATen/native/quantized/cpu/fbgemm_utils.h>
#include <torch/library.h>
#include <c10/util/irange.h>

#define C0 0.28209479177387814f
#define C1 0.4886025119029199f
#define C20 1.0925484305920792f
#define C21 -1.0925484305920792f
#define C22 0.31539156525252005f
#define C23 -1.0925484305920792f
#define C24 0.5462742152960396f
#define C30 -0.5900435899266435f
#define C31 2.890611442640554f
#define C32 -0.4570457994644658f
#define C33 0.3731763325901154f
#define C34 -0.4570457994644658f
#define C35 1.445305721320277f
#define C36 -0.5900435899266435f
#define C40 2.5033429417967046f
#define C41 -1.7701307697799304f
#define C42 0.9461746957575601f
#define C43 -0.6690465435572892f
#define C44 0.10578554691520431f
#define C45 -0.6690465435572892f
#define C46 0.47308734787878004f
#define C47 -1.7701307697799304f
#define C48 0.6258357354491761f


__device__ __constant__ float c_origin[3];


__device__ __forceinline__
void leakyRelu(half* x, int size, half negative_slope)
{
    #pragma unroll
    for (int i = 0; i < size; i++)
    {
        if (__hlt(x[i], __float2half(0.0f))) 
        {
            x[i] = __hmul(x[i], __float2half(0.01f));
        }
    }
}

__device__ __forceinline__
float3 Sigmoid(float3 x)
{
    x.x = 1.0f / (1.0f + expf(-1.0f * x.x));
    x.y = 1.0f / (1.0f + expf(-1.0f * x.y));
    x.z = 1.0f / (1.0f + expf(-1.0f * x.z));
    return x;
}


inline __device__
float3 inference_SH(const float3 rays_d, __half* coeffi)
{
    __half color[3];

    #pragma unroll
    for (int i=0; i<3; i++)
    {
        color[i] = __hmul(__float2half(C0), coeffi[i]);
    }
    __half x = __float2half(rays_d.x), y = __float2half(rays_d.y), z = __float2half(rays_d.z);
    
    #pragma unroll
    for (int i=0; i<3; i++)
    {
        __half item1 = __hmul(__hmul(__float2half(C1), y), coeffi[3 + i]);
        __half item2 = __hmul(__hmul(__float2half(C1), z), coeffi[6 + i]);
        __half item3 = __hmul(__hmul(__float2half(C1), x), coeffi[9 + i]);
        color[i] = __hsub(__hadd(__hsub(color[i], item1), item2), item3);
    }

    __half xx = __hmul(x,x);
    __half yy = __hmul(y,y);
    __half zz = __hmul(z,z);
    __half xy = __hmul(x,y);
    __half yz = __hmul(y,z);
    __half xz = __hmul(x,z);

    __half temp = __hsub(__hsub(__hmul(__float2half(2.0f), zz), xx), yy);
    __half temp2 = __hsub(xx, yy);

    #pragma unroll
    for (int i=0; i<3; i++)
    {
        __half item1 = __hmul(__hmul(__float2half(C20), xy), coeffi[12 + i]);
        __half item2 = __hmul(__hmul(__float2half(C21), yz), coeffi[15 + i]);
        __half item3 = __hmul(__hmul(__float2half(C22), temp), coeffi[18 + i]);
        __half item4 = __hmul(__hmul(__float2half(C23), xz), coeffi[21 + i]);
        __half item5 = __hmul(__hmul(__float2half(C24), temp2), coeffi[24 + i]);
        color[i] = __hadd(color[i], item1);
        color[i] = __hadd(color[i], item2);
        color[i] = __hadd(color[i], item3);
        color[i] = __hadd(color[i], item4);
        color[i] = __hadd(color[i], item5);
    }

    return make_float3(__half2float(color[0]),__half2float(color[1]),__half2float(color[2]));
}


__global__ void inference_mlp_kernel(
    int max_samples_per_ray,
    float inverse_far,
    short* netIdxs, 
    float* transparency,
    float trans_stop, 
    const int* query_pixel_indices, // cur pixel -> ori pixel 
    const int* pixel_strats, 
    const float* group_centers,
    const float* sample_points,
    const float3* ray_directions,
    const __half* network_params,
    float4* sample_results) // N x 4
{
    int netIdx = netIdxs[blockIdx.x];

    if (netIdx == -1) return;
    
    int cur_task_id = pixel_strats[blockIdx.x] * max_samples_per_ray + threadIdx.x;

    constexpr float frequency_bands[10] = {1., 2., 4., 8., 16., 32., 64., 128., 256., 512.};
    constexpr int deg_SH = 2;
    constexpr int hidden_layer = 64;
    constexpr int embedding_L = 10;
    constexpr int input_layer = 63;
    constexpr int output_layer = 28;
    constexpr int param_size = 22556;

    __shared__ float center[3];
    if (threadIdx.x < 3)
    {
        center[threadIdx.x] = group_centers[netIdx*3+threadIdx.x];
    }

    __shared__ half network_cache[param_size];

    int load_idx = threadIdx.x;
    int network_offset = netIdx * param_size;
    while (load_idx < param_size) {
        network_cache[load_idx] = network_params[network_offset+load_idx];
        load_idx += blockDim.x;
    }
    __syncthreads();
    

    half layer0[hidden_layer*2];

    while (cur_task_id < pixel_strats[blockIdx.x+1]*max_samples_per_ray) {

        int oriPixel = query_pixel_indices[cur_task_id / max_samples_per_ray];

        if (transparency[oriPixel] < trans_stop) 
        {
            cur_task_id += blockDim.x;
            continue;
        }


        int ori_task_id = oriPixel * max_samples_per_ray + cur_task_id % max_samples_per_ray;

        float point[3];
        #pragma unroll
        for (int i=0; i<3; i++)
        {
            point[i] = (sample_points[ori_task_id*3 + i] - center[i]) * inverse_far;
        }

        layer0[0] = __float2half(point[0]);
        layer0[1] = __float2half(point[1]);
        layer0[2] = __float2half(point[2]);


        int param_index = 3;
        #pragma unroll
        for (int i = 0; i < embedding_L; i++)
        {
            layer0[param_index] = __float2half(__sinf(frequency_bands[i] * point[0]));
            layer0[param_index + 1] = __float2half(__sinf(frequency_bands[i] * point[1]));
            layer0[param_index + 2] = __float2half(__sinf(frequency_bands[i] * point[2]));
            layer0[param_index + 3] = __float2half(__cosf(frequency_bands[i] * point[0]));
            layer0[param_index + 4] = __float2half(__cosf(frequency_bands[i] * point[1]));
            layer0[param_index + 5] = __float2half(__cosf(frequency_bands[i] * point[2]));
            param_index += 6;
        }


        param_index = 0;

        //layer0->layer0
        #pragma unroll
        for (int i = hidden_layer; i < hidden_layer*2; ++i) {
            layer0[i] = network_cache[param_index++];
        }

        #pragma unroll
        for (int i = 0; i < input_layer; ++i) {
            #pragma unroll
            for (int j = hidden_layer; j < hidden_layer*2; ++j) {
                layer0[j] = __hfma(layer0[i], network_cache[param_index++], layer0[j]);
            }
        }

        leakyRelu(layer0+hidden_layer, hidden_layer, __float2half(0.01f));

        //layer0->layer2
        #pragma unroll
        for (int i = 0; i < hidden_layer; ++i) {
            layer0[i] = network_cache[param_index++];
        }

        #pragma unroll
        for (int i = hidden_layer; i < hidden_layer*2; ++i) {
            #pragma unroll
            for (int j = 0; j < hidden_layer; ++j) {
                layer0[j] = __hfma(layer0[i], network_cache[param_index++], layer0[j]);
            }
        }
        leakyRelu(layer0, hidden_layer, __float2half(0.01f));

        //layer2->layer3
        #pragma unroll
        for (int i = hidden_layer; i < hidden_layer*2; ++i) {
            layer0[i] = network_cache[param_index++];
        }

        #pragma unroll
        for (int i = 0; i < hidden_layer; ++i) {
            #pragma unroll
            for (int j = hidden_layer; j < hidden_layer*2; ++j) {
                layer0[j] = __hfma(layer0[i], network_cache[param_index++], layer0[j]);
            }
        }
        leakyRelu(layer0+hidden_layer, hidden_layer, __float2half(0.01f));

        //layer3->layer4
        #pragma unroll
        for (int i = 0; i < hidden_layer; ++i) {
            layer0[i] = network_cache[param_index++];
        }

        #pragma unroll
        for (int i = hidden_layer; i < hidden_layer*2; ++i) {
            #pragma unroll
            for (int j = 0; j < hidden_layer; ++j) {
                layer0[j] = __hfma(layer0[i], network_cache[param_index++], layer0[j]);
            }
        }
        leakyRelu(layer0, hidden_layer, __float2half(0.01f));

        //layer4->layer5
        #pragma unroll
        for (int i = hidden_layer; i < hidden_layer*2; ++i) {
            layer0[i] = network_cache[param_index++];
        }

        #pragma unroll
        for (int i = 0; i < hidden_layer; ++i) {
            #pragma unroll
            for (int j = hidden_layer; j < hidden_layer*2; ++j) {
                layer0[j] = __hfma(layer0[i], network_cache[param_index++], layer0[j]);
            }
        }
        leakyRelu(layer0+hidden_layer, hidden_layer, __float2half(0.01f));

        // layer5->layer6
        #pragma unroll
        for (int i = 0; i < output_layer; ++i) {
            layer0[i] = network_cache[param_index++];
        }

        #pragma unroll
        for (int i = hidden_layer; i < hidden_layer*2; ++i) {
            #pragma unroll
            for (int j = 0; j < output_layer; ++j) {
                layer0[j] = __hfma(layer0[i], network_cache[param_index++], layer0[j]);
            }
        }


        float sigma = __half2float(layer0[output_layer-1]);
        sigma = fmaxf(sigma, 0);

        float3 raw = inference_SH(normalize(ray_directions[oriPixel]), layer0);

        float3 rgb = Sigmoid(raw);

        sample_results[ori_task_id] = make_float4(rgb, sigma);

        cur_task_id += blockDim.x;
    }
}


__host__ 
void inference_mlp(
    int max_samples_per_ray,
    int num_block, int num_thread,
    float inverse_far, float trans_stop,
    at::Tensor netIdxs,
    at::Tensor transparency,
    at::Tensor query_pixel_indices,
    at::Tensor pixel_strats,
    at::Tensor group_centers,
    at::Tensor samples,
    at::Tensor rays_d,
    at::Tensor params,
    at::Tensor sample_results)
{

    inference_mlp_kernel<<<num_block, num_thread>>>(
        max_samples_per_ray, inverse_far,
        netIdxs.contiguous().data_ptr<short>(),
        transparency.contiguous().data_ptr<float>(), trans_stop, 
        query_pixel_indices.contiguous().data_ptr<int>(),
        pixel_strats.contiguous().data_ptr<int>(),
        group_centers.contiguous().data_ptr<float>(),
        samples.contiguous().data_ptr<float>(),
        (float3*)rays_d.contiguous().data_ptr<float>(),
        (__half*)params.contiguous().data_ptr<at::Half>(),
        (float4*)sample_results.contiguous().data_ptr<float>());

    AT_CUDA_CHECK(cudaGetLastError());

    return;
}
