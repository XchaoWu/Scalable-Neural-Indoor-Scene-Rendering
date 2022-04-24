#include "macros.h"
#include "cutil_math.h"
#include "cuda_utils.h"
#include <ATen/ATen.h>
#include <stdint.h>
#include <ATen/TensorAccessor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <torch/extension.h>
#include "interpolation.h"

__device__ static float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
        } while (assumed != old);
    
    return __int_as_float(old);
}

template<class T>
inline __device__ T transparency_interpolation(
    const T* voxels,
    const float3 tile_center,
    const float tile_size,
    const float transparency,
    const int num_voxel,
    const float voxel_size,
    const float3 pts,
    float* T_voxels)
{

    float3 tile_corner = tile_center - tile_size / 2.0f;
    // overlap 1 voxel
    float3 fidx = (pts - tile_corner) / voxel_size + 0.5f;
    int x0 = (int)fidx.x, y0 = (int)fidx.y, z0 = (int)fidx.z;
    int x1 = x0 + 1, y1 = y0 + 1, z1 = z0 + 1;
    float x = fidx.x - x0, y = fidx.y - y0, z = fidx.z - z0;


    assert(x0 >= 0 && y0 >= 0 && z0 >= 0 && 
           x1 < num_voxel && y1 < num_voxel && z1 < num_voxel);

    float w000 = (1-x) * (1-y) * (1-z);
    float w001 = (1-x) * (1-y) * z;
    float w010 = (1-x) * y * (1-z);
    float w011 = (1-x) * y * z;
    float w100 = x * (1-y) * (1-z);
    float w101 = x * (1-y) * z;
    float w110 = x * y * (1-z);
    float w111 = x * y * z;

    int pow_num_voxel = num_voxel * num_voxel;

    float t000 = transparency * w000;
    float t001 = transparency * w001;
    float t010 = transparency * w010;
    float t011 = transparency * w011;
    float t100 = transparency * w100;
    float t101 = transparency * w101;
    float t110 = transparency * w110;
    float t111 = transparency * w111;

    int v000 = x0 * pow_num_voxel + y0 * num_voxel + z0;
    int v001 = v000 + 1;
    int v010 = v000 + num_voxel;
    int v011 = v010 + 1;
    int v100 = v000 + pow_num_voxel;
    int v101 = v100 + 1;
    int v110 = v100 + num_voxel;
    int v111 = v110 + 1;

    atomicMax(T_voxels+v000, t000);
    atomicMax(T_voxels+v001, t001);
    atomicMax(T_voxels+v010, t010);
    atomicMax(T_voxels+v011, t011);
    atomicMax(T_voxels+v100, t100);
    atomicMax(T_voxels+v101, t101);
    atomicMax(T_voxels+v110, t110);
    atomicMax(T_voxels+v111, t111);
    // T_voxels[v000] = T_voxels[v000] < t000? t000:T_voxels[v000];
    // T_voxels[v001] = T_voxels[v001] < t001? t001:T_voxels[v001];
    // T_voxels[v010] = T_voxels[v010] < t010? t010:T_voxels[v010];
    // T_voxels[v011] = T_voxels[v011] < t011? t011:T_voxels[v011];
    // T_voxels[v100] = T_voxels[v100] < t100? t100:T_voxels[v100];
    // T_voxels[v101] = T_voxels[v101] < t101? t101:T_voxels[v101];
    // T_voxels[v110] = T_voxels[v110] < t110? t110:T_voxels[v110];
    // T_voxels[v111] = T_voxels[v111] < t111? t111:T_voxels[v111];

    return voxels[v000] * w000 + voxels[v001] * w001 + 
           voxels[v010] * w010 + voxels[v011] * w011 + 
           voxels[v100] * w100 + voxels[v101] * w101 + 
           voxels[v110] * w110 + voxels[v111] * w111;
           
    
}


__global__ 
void transparency_statistic_cuda_kernel(
    const float* rays_start,
    const float* rays_dir,
    const int num_voxel,
    const float* center,
    const float tile_size,
    const float voxel_size,
    const float sample_step,
    float* voxels, // N x N x N
    short* nodes, // N x N x N 
    float* T_voxels, // N x N x N 
    int base_jobs, int add_num)
{
    int cur_thread = threadIdx.x + blockIdx.x * blockDim.x;
    int task_num,start_task_idx;

    if (cur_thread >= add_num){
        task_num = base_jobs;
        start_task_idx = cur_thread * base_jobs + add_num;
    }else{
        task_num = base_jobs + 1;
        start_task_idx = cur_thread * (base_jobs + 1);
    }

    float3 tile_center = make_float3(center[0],center[1],center[2]);
    float3 dilate_corner = tile_center - tile_size/2.0f - voxel_size;

    for (int i=0; i<task_num; i++)
    {
        int cur_task_idx = start_task_idx + i;
        float3 rays_o = make_float3(rays_start[cur_task_idx*3],
                                    rays_start[cur_task_idx*3+1],
                                    rays_start[cur_task_idx*3+2]);
        float3 rays_d = make_float3(rays_dir[cur_task_idx*3],
                                    rays_dir[cur_task_idx*3+1],
                                    rays_dir[cur_task_idx*3+2]);
        
        float2 bound = RayAABBIntersection(rays_o, rays_d, tile_center, tile_size/2.0f);

        float zval = bound.x;
        float transparency = 1.0f;

        while(zval < bound.y)
        {
            float3 pts = rays_o + zval * rays_d;

            int3 pidx = make_int3((pts-dilate_corner)/voxel_size);
            int nidx = pidx.x * num_voxel * num_voxel + pidx.y * num_voxel + pidx.z;

            if (nodes[nidx] != 1)
            {
                zval += sample_step;
                continue;
            }

            float sigma = transparency_interpolation(voxels, tile_center, tile_size, transparency, num_voxel, voxel_size, pts, T_voxels);

            float alpha = 1.0f - expf(-1.0f * sigma * sample_step);

            transparency = transparency * (1.0f - alpha);

            if (transparency < 0.00000001f) break;

            zval = zval + sample_step;
        }
        
    }
}

__global__ 
void dilate_boundary_cuda_kernel(
    short* nodes_flag,
    short* new_nodes_flag,
    int num_voxel,    
    int base_jobs, int add_num)
{
    int cur_thread = threadIdx.x + blockIdx.x * blockDim.x;
    int task_num,start_task_idx;

    if (cur_thread >= add_num){
        task_num = base_jobs;
        start_task_idx = cur_thread * base_jobs + add_num;
    }else{
        task_num = base_jobs + 1;
        start_task_idx = cur_thread * (base_jobs + 1);
    }

    int t1 = num_voxel * num_voxel;

    for (int i=0; i<task_num; i++)
    {
        int cur_task_idx = start_task_idx + i;

        new_nodes_flag[cur_task_idx] = nodes_flag[cur_task_idx];

        if (nodes_flag[cur_task_idx] != -1) continue; 

        int x = cur_task_idx / t1;
        int temp = cur_task_idx % t1;
        int y = temp / num_voxel;
        int z = temp % num_voxel;

        int3 start_pos = make_int3(x-1,y-1,z-1);
        start_pos = clamp(start_pos, 0, num_voxel-1);
        int3 end_pos = make_int3(x+1,y+1,z+1);
        end_pos = clamp(end_pos, 0, num_voxel-1);

        for (int _x=start_pos.x;_x<=end_pos.x;_x++)
        {
            for (int _y=start_pos.y;_y<=end_pos.y;_y++)
            {
                for (int _z=start_pos.z;_z<=end_pos.z;_z++)
                {
                    int index = _x * t1 + _y * num_voxel + _z;
                    if (nodes_flag[index] == 1)
                    {
                        new_nodes_flag[cur_task_idx] = 0;
                        break;
                    }
                }
            }
        }

    }
}

void dilate_boundary_cuda(
    at::Tensor &nodes_flag,
    int num_voxel)
{
    int total_voxels = num_voxel * num_voxel * num_voxel;

    unsigned int n_threads, n_blocks;

    n_threads = 512;
    n_blocks = min(65535, (total_voxels + n_threads - 1) / n_threads);

    int add_num = total_voxels % (n_blocks * n_threads);
    int base_jobs = total_voxels / (n_blocks * n_threads);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();


    short* new_nodes_flag;
    cudaMalloc((void**)&new_nodes_flag, sizeof(short)*total_voxels);

    dilate_boundary_cuda_kernel<<<n_blocks, n_threads, 0, stream>>>(
        nodes_flag.contiguous().data_ptr<short>(), 
        new_nodes_flag, num_voxel, base_jobs, add_num);
    
    cudaMemcpy(nodes_flag.contiguous().data_ptr<short>(),
               new_nodes_flag, sizeof(short)*total_voxels, cudaMemcpyDeviceToDevice);

    cudaFree(new_nodes_flag);

    AT_CUDA_CHECK(cudaGetLastError());
    return;  
}

__host__
void transparency_statistic_cuda(
    const at::Tensor rays_start,
    const at::Tensor rays_dir,
    const int num_voxel,
    const at::Tensor tile_center,
    const float tile_size,
    const float voxel_size,
    const float sample_step,
    at::Tensor voxels,
    at::Tensor nodes,
    at::Tensor &T_voxels)
{
    int batchSize = rays_start.size(0);

    unsigned int n_threads, n_blocks;

    n_threads = 512;
    n_blocks = min(65535, (batchSize + n_threads - 1) / n_threads);

    int add_num = batchSize % (n_blocks * n_threads);
    int base_jobs = batchSize / (n_blocks * n_threads);

    transparency_statistic_cuda_kernel<<<n_blocks, n_threads>>>(
        rays_start.contiguous().data_ptr<float>(),
        rays_dir.contiguous().data_ptr<float>(),
        num_voxel, tile_center.contiguous().data_ptr<float>(),
        tile_size, voxel_size, sample_step,
        voxels.contiguous().data_ptr<float>(),
        nodes.contiguous().data_ptr<short>(),
        T_voxels.contiguous().data_ptr<float>(),
        base_jobs, add_num);

    AT_CUDA_CHECK(cudaGetLastError());
    return;  
}