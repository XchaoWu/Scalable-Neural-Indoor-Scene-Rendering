#include "volume_rendering.h"

#define sample_per_ray 16 

__global__
void integrate_points_kernel(
    const float4* sample_results, 
    const float* intergrate_steps, 
    float* transparency, float early_step,
    float3* colors_buffer,
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

    
    while (cur_task_idx < end_task_idx)
    {
        float T = transparency[cur_task_idx];

        if (T < early_step) 
        {
            cur_task_idx++;
            continue;
        }

        float3 color = make_float3(0.0f,0.0f,0.0f);

        const float4* cur_sample_results = sample_results + cur_task_idx * sample_per_ray;
        const float* cur_intergrate_steps = intergrate_steps + cur_task_idx * sample_per_ray;


        #pragma unroll 
        for (int i=0; i<sample_per_ray; i++)
        {
            float4 vec = cur_sample_results[i];
            float3 rgb = make_float3(vec.x, vec.y, vec.z);

            float alpha = 1.0f - __expf(-1.0f * vec.w * cur_intergrate_steps[i]);
            color = color + alpha * T * rgb;
            T = T * (1-alpha);
        }

        transparency[cur_task_idx] = T; 
        colors_buffer[cur_task_idx] += color;

        cur_task_idx++;
    }
}

__host__
void integrate_points(
    at::Tensor sample_results,
    at::Tensor intergrate_steps,
    int num_thread, float early_step, 
    at::Tensor &transparency,
    at::Tensor &colors_buffer)
{
    int H = colors_buffer.size(0);
    int W = colors_buffer.size(1);
    int numPixel = H * W;

    int num_block = min(65535, (numPixel + num_thread - 1) / num_thread);
    int base_num = numPixel / (num_block * num_thread);
    int extra_num = numPixel - base_num * (num_block * num_thread);

    integrate_points_kernel<<<num_block, num_thread>>>(
        (float4*)sample_results.contiguous().data_ptr<float>(),
        intergrate_steps.contiguous().data_ptr<float>(),
        transparency.contiguous().data_ptr<float>(), early_step,
        (float3*)colors_buffer.contiguous().data_ptr<float>(),
        base_num, extra_num);

    AT_CUDA_CHECK(cudaGetLastError());

    return;

}

__host__
void frame_memcpy(long frame_pointer, const at::Tensor &frame)
{
    int H = frame.size(0);
    int W = frame.size(1);
    int numPixel = H * W;
    float *frame_map = (float*)frame_pointer;
    cudaMemcpy(frame_map, frame.contiguous().data_ptr<float>(), sizeof(float)*numPixel*3, cudaMemcpyDeviceToDevice);
}