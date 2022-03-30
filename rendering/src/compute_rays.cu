#include "compute_rays.h"


__device__ __constant__ float c_c2w[9];
__device__ __constant__ float c_intrinsic[3];

__device__ __forceinline__ 
void get_direction(int _x, int _y, float *direction)
{
    float x = ( (float)_x - c_intrinsic[0]) * c_intrinsic[2];
    float y = ( (float)_y - c_intrinsic[1]) * c_intrinsic[2];

    direction[0] = c_c2w[0] * x + c_c2w[1] * y + c_c2w[2];
    direction[1] = c_c2w[3] * x + c_c2w[4] * y + c_c2w[5];
    direction[2] = c_c2w[6] * x + c_c2w[7] * y + c_c2w[8];
}

__global__ 
void compute_rays_kernel(
    float* rays_d, int W,
    int extra_num, int base_num)
{
    int cur_thread = threadIdx.x + blockIdx.x * blockDim.x;

    int task_num = base_num, cur_task_idx = 0;
    if (cur_thread < extra_num)
    {
        task_num += 1;
    }else{
        cur_task_idx += extra_num;
    }
    cur_task_idx += cur_thread * task_num;
    int end_task_idx = cur_task_idx + task_num;

    while (cur_task_idx < end_task_idx)
    {
        int py = cur_task_idx / W;
        int px = cur_task_idx - py * W;

        get_direction(px, py, rays_d + 3 * cur_task_idx);

        cur_task_idx++;
    }
}

__host__ 
void compute_rays(
    at::Tensor intrinsic, 
    at::Tensor c2w_rotation,
    int num_thread,
    at::Tensor &rays_d)
{   
    int H = rays_d.size(0);
    int W = rays_d.size(1);
    int numPixel = H * W;
    int num_block = min(65535, (numPixel + num_thread - 1) / num_thread);
    int base_num = numPixel / (num_block * num_thread);
    int extra_num = numPixel - base_num * (num_block * num_thread);
    cudaMemcpyToSymbol(c_c2w, c2w_rotation.contiguous().data_ptr<float>(), sizeof(float)*9, 0, cudaMemcpyDeviceToDevice);
    cudaMemcpyToSymbol(c_intrinsic, intrinsic.contiguous().data_ptr<float>(), sizeof(float)*3, 0, cudaMemcpyDeviceToDevice);
    compute_rays_kernel<<<num_block, num_thread>>>(
        rays_d.contiguous().data_ptr<float>(), 
        W, extra_num, base_num);

    AT_CUDA_CHECK(cudaGetLastError());

    return;
}

