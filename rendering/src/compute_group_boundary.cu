#include "compute_group_boundary.h"


__global__ 
void compute_group_boundary_kernel(
    short* netIdx, int H, int W, 
    bool* boundary, int ksize,
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

    int radius = ksize / 2;

    while (cur_task_idx < end_task_idx)
    {
        int py = cur_task_idx / W;
        int px = cur_task_idx - py * W;

        short cur_netIdx = netIdx[cur_task_idx];

        int min_x = max(px - radius, 0);
        int max_x = min(px + radius, W-1);
        int min_y = max(py - radius, 0);
        int max_y = min(py + radius, H-1);

        for (int y=min_y; y<=max_y && !boundary[cur_task_idx]; y++)
        {
            for(int x=min_x; x<=max_x; x++)
            {
                int idx = y * W + x;
                if (netIdx[idx] != cur_netIdx)
                {
                    boundary[cur_task_idx] = true;
                    break;
                }
            }

        }

        cur_task_idx++;
    }
}


__host__
void compute_group_boundary(
    at::Tensor netIdx, 
    at::Tensor &boundary,
    int ksize,
    int num_thread)
{
    int H = netIdx.size(0);
    int W = netIdx.size(1);
    int numPixel = H * W;
    int num_block = min(65535, (numPixel + num_thread - 1) / num_thread);
    int base_num = numPixel / (num_block * num_thread);
    int extra_num = numPixel - base_num * (num_block * num_thread);
    compute_group_boundary_kernel<<<num_block, num_thread>>>(
        netIdx.contiguous().data_ptr<short>(), H, W, 
        boundary.contiguous().data_ptr<bool>(), 
        ksize, extra_num, base_num);

    AT_CUDA_CHECK(cudaGetLastError());

    return;
}