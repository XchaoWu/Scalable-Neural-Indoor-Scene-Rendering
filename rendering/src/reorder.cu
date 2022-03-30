
#include "reorder.h"

#include <thrust/gather.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <ATen/cuda/CUDABlas.h>

using namespace torch::indexing;

__host__
int sort_by_key(at::Tensor &keys_tensor, at::Tensor &values_tensor, at::Tensor &starts_tensor)
{
    int length = keys_tensor.size(0);
    short* keys = keys_tensor.data_ptr<short>();
    int* values = values_tensor.data_ptr<int>();
    int* starts = starts_tensor.data_ptr<int>();

    thrust::sort_by_key(thrust::device, keys, keys + length, values);

    thrust::pair<short*, int*> new_end = thrust::unique_by_key(thrust::device, keys, keys + length, starts);
    return new_end.second - starts;
}

__global__ 
void assign_pixels_kernel(
    const short* netIdxs,
    const int* pixel_starts,
    const int* block_starts,
    short* new_netIdxs,
    int* new_pixel_starts,
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
        short netIdx = netIdxs[cur_task_idx];
        int start_loc = block_starts[cur_task_idx];
        int pixel_start = pixel_starts[cur_task_idx];
        int num_pixels_per_net = pixel_starts[cur_task_idx+1] - pixel_starts[cur_task_idx];
         
        int num_block = block_starts[cur_task_idx+1] - start_loc;
        int base_pixels = num_pixels_per_net / num_block;
        int add_num = num_pixels_per_net - base_pixels * num_block;

        for (int i=0; i<num_block; i++)
        {
            new_netIdxs[start_loc+i] = netIdx;
            new_pixel_starts[start_loc+i] = pixel_start;
            if (i < add_num) pixel_start += (base_pixels+1);
            else pixel_start += base_pixels;
        }

        cur_task_idx++;
    }
}

__host__
void assign_pixels(at::Tensor netIdxs, // n-1
                   at::Tensor pixel_starts, // n 
                   at::Tensor block_starts, // n
                   int num_thread,
                   at::Tensor &new_netIdxs,
                   at::Tensor &new_pixel_starts)
{

    int numNet = netIdxs.size(0);
    int num_block = min(65535, (numNet + num_thread - 1) / num_thread);
    int base_num = numNet / (num_block * num_thread);
    int extra_num = numNet - base_num * (num_block * num_thread);

    assign_pixels_kernel<<<num_block, num_thread>>>(
        netIdxs.contiguous().data_ptr<short>(),
        pixel_starts.contiguous().data_ptr<int>(),
        block_starts.contiguous().data_ptr<int>(),
        new_netIdxs.contiguous().data_ptr<short>(),
        new_pixel_starts.contiguous().data_ptr<int>(),
        base_num, extra_num);
    
    AT_CUDA_CHECK(cudaGetLastError());

    return;
}