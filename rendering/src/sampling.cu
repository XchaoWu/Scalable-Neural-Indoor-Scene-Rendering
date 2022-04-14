#include "sampling.h"

__device__ __constant__ float c_origin[3];

__global__
void inverse_z_sampling_kernel(
    const float* rays_d,
    const float* inverse_near,
    const float* inverse_bound, 
    int offset, int sample_per_ray,
    float* sample_points, float* integrate_steps,
    int base_num, int extra_num)
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

    //  points
    while (cur_task_idx < end_task_idx)
    {
        int pixelIdx = cur_task_idx / sample_per_ray;
        int pointIdx = cur_task_idx % sample_per_ray;

        float zval = __fdiv_rn(1.0f, (offset + pointIdx) * inverse_bound[pixelIdx] + inverse_near[pixelIdx]);
        float next_zval = __fdiv_rn(1.0f, (offset + pointIdx + 1) * inverse_bound[pixelIdx] + inverse_near[pixelIdx]);

        integrate_steps[cur_task_idx] = fmax(next_zval - zval, 0);
        
        #pragma unroll 
        for (int i=0; i<3; i++)
        {
            sample_points[cur_task_idx*3+i] = c_origin[i] + zval * rays_d[pixelIdx*3+i];
        }

        cur_task_idx++;
    }
}


__host__ 
void inverse_z_sampling(  
    at::Tensor origin,
    at::Tensor rays_d,
    at::Tensor inverse_near,
    at::Tensor inverse_bound,
    int num_thread,
    int offset, int sample_per_ray,
    at::Tensor &sample_points,
    at::Tensor &integrate_steps)
{
    int H = inverse_near.size(0);
    int W = inverse_near.size(1);
    int numPixel = H * W;

    int num_block = min(65535, (numPixel * sample_per_ray + num_thread - 1) / num_thread);
    int base_num = numPixel * sample_per_ray / (num_block * num_thread);
    int extra_num = numPixel * sample_per_ray - base_num * (num_block * num_thread);

    cudaMemcpyToSymbol( c_origin, origin.contiguous().data_ptr<float>(), sizeof(float)*3, 0, cudaMemcpyDeviceToDevice);

    inverse_z_sampling_kernel<<<num_block,num_thread>>>(
        rays_d.contiguous().data_ptr<float>(),
        inverse_near.contiguous().data_ptr<float>(),
        inverse_bound.contiguous().data_ptr<float>(),
        offset, sample_per_ray, 
        sample_points.contiguous().data_ptr<float>(),
        integrate_steps.contiguous().data_ptr<float>(),
        base_num, extra_num);

    AT_CUDA_CHECK(cudaGetLastError());

    return;
}