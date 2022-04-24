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

__global__ 
void Sample_reflection_cuda_kernel(
    const float* rays_start,
    const float* rays_dir,
    const short* node_flags, // W x H x D (x,y,z) flatten  
    const float* voxels, // W x H x D x 4
    const float3 center,
    const float tile_size,
    const int Nsamples,
    const float far, 
    const int num_voxel,
    const float T_th,
    float* z_vals,
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

    float half_size = tile_size * 0.5f;
    float voxel_size = tile_size / num_voxel;
    float3 min_corner = center - half_size;

    for (int i=0; i<task_num; i++)
    {
        int cur_task_idx = start_task_idx + i;
        float* cur_z_vals = z_vals + cur_task_idx * Nsamples;
        float3 rays_o = make_float3(rays_start[cur_task_idx*3],
                                    rays_start[cur_task_idx*3+1],
                                    rays_start[cur_task_idx*3+2]);
        float3 rays_d = make_float3(rays_dir[cur_task_idx*3],
                                    rays_dir[cur_task_idx*3+1],
                                    rays_dir[cur_task_idx*3+2]);

        float2 bound = RayAABBIntersection(rays_o, rays_d, center, half_size);
        if (bound.x == -1 && bound.y == -1) continue;

        float3 rays_o_local = rays_o + bound.x * rays_d - min_corner;
        rays_o_local = clamp(rays_o_local, 0.0f, tile_size-0.000001f);
        int3 current_voxel = make_int3(rays_o_local / voxel_size);

        current_voxel = clamp(current_voxel, 0, num_voxel-1);
        int3 step = signf(rays_d);
        float3 next_boundary = make_float3(current_voxel + step) * voxel_size;
        if (step.x < 0) next_boundary.x += voxel_size;
        if (step.y < 0) next_boundary.y += voxel_size;
        if (step.z < 0) next_boundary.z += voxel_size;

        float3 tMax = safe_divide(next_boundary-rays_o_local, rays_d);
        float3 tDelta = safe_divide(make_float3(step)*voxel_size, rays_d);   

        float transparency = 1.0f;

        float near = -1;
        while(true)
        {
            int vidx = get_index(num_voxel, num_voxel, num_voxel, &current_voxel.x);
            if (node_flags[vidx] == 1)
            {
                float3 voxel_center = min_corner + make_float3(current_voxel) * voxel_size + 0.5f * voxel_size;

                float2 voxel_bound = RayAABBIntersection(rays_o, rays_d, voxel_center, 0.5f * voxel_size);

                if (voxel_bound.x != -1 && voxel_bound.y != -1)
                {
                    float3 pts = rays_o + voxel_bound.x * rays_d;
                    float4 rgba = voxels_interpolation((float4*)voxels, center, tile_size, num_voxel,
                                                            voxel_size, pts, OVERLAP);
                    float sigma = rgba.w;
                    float dist  = voxel_bound.y - voxel_bound.x;
                    float alpha = 1.0f - expf(-1.0f * sigma * dist);
                    transparency = transparency * (1.0f - alpha);

                    if (transparency <= T_th)
                    {
                        near = voxel_bound.y;
                        break;
                    }
                }
            }

            if (tMax.x < tMax.y){
                if (tMax.x < tMax.z)
                {
                    current_voxel.x += step.x;
                    tMax.x += tDelta.x;
                }else{
                    current_voxel.z += step.z;
                    tMax.z += tDelta.z;
                }
            }else{
                if (tMax.y < tMax.z)
                {
                    current_voxel.y += step.y;
                    tMax.y += tDelta.y;
                }else{
                    current_voxel.z += step.z;
                    tMax.z += tDelta.z;
                }
            }

            if (current_voxel.x < 0 || 
                current_voxel.y < 0 || 
                current_voxel.z < 0 || 
                current_voxel.x >= num_voxel || 
                current_voxel.y >= num_voxel || 
                current_voxel.z >= num_voxel) break;

        }
        if (near == -1) continue;

        inverse_z_sample_bound(cur_z_vals, near, far, Nsamples);

    }
}

void Sample_reflection_cuda(
    const at::Tensor rays_start,
    const at::Tensor rays_dir,
    const at::Tensor node_flags,
    const at::Tensor voxels,
    const float cx,
    const float cy,
    const float cz,
    const float tile_size,
    const float far,
    const int num_voxel,
    const float T_th, 
    at::Tensor &z_vals)
{
    int batchSize = rays_start.size(0);
    int Nsamples = z_vals.size(1);
    float3 center = make_float3(cx,cy,cz);

    unsigned int n_threads, n_blocks;

    n_threads = 512;
    n_blocks = min(65535, (batchSize + n_threads - 1) / n_threads);

    int add_num = batchSize % (n_blocks * n_threads);
    int base_jobs = batchSize / (n_blocks * n_threads);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    Sample_reflection_cuda_kernel<<<n_blocks, n_threads, 0, stream>>>(
        rays_start.contiguous().data_ptr<float>(),
        rays_dir.contiguous().data_ptr<float>(),
        node_flags.contiguous().data_ptr<short>(),
        voxels.contiguous().data_ptr<float>(),
        center, tile_size, Nsamples, far, num_voxel, T_th,
        z_vals.contiguous().data_ptr<float>(),
        base_jobs, add_num);

    AT_CUDA_CHECK(cudaGetLastError());
    return;  
}


__global__ 
void Sample_sparse_cuda_kernel(
    const float* rays_start,
    const float* rays_dir,
    const short* node_flags, // W x H x D (x,y,z) flatten  
    const float3 center,
    const float tile_size,
    const int Nsamples,
    const int num_voxel,
    float* z_vals, // B x Nsamples  z_vals
    float* dists, // B x Nsamples
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

    float half_size = tile_size * 0.5f;
    float voxel_size = tile_size / num_voxel;
    float3 min_corner = center - half_size;

    for (int i=0; i<task_num; i++)
    {
        int cur_task_idx = start_task_idx + i;
        float* cur_z_vals = z_vals + cur_task_idx * Nsamples;
        float* cur_dists = dists + cur_task_idx * Nsamples;

        float3 rays_o = make_float3(rays_start[cur_task_idx*3],
                                    rays_start[cur_task_idx*3+1],
                                    rays_start[cur_task_idx*3+2]);
        float3 rays_d = make_float3(rays_dir[cur_task_idx*3],
                                    rays_dir[cur_task_idx*3+1],
                                    rays_dir[cur_task_idx*3+2]);
        float2 near_far = RayAABBIntersection(rays_o, rays_d, center, half_size);

        if (near_far.x == -1 && near_far.y == -1) continue;

        float near = near_far.x;
        float far = near_far.y;

        float total_length = 0.0f; 

        float3 rays_o_local = rays_o + near * rays_d - min_corner;
        rays_o_local = clamp(rays_o_local, 0.0f, tile_size-0.000001f);
        int3 current_voxel = make_int3(rays_o_local / voxel_size);

        current_voxel = clamp(current_voxel, 0, num_voxel-1);
        int3 step = signf(rays_d);
        float3 next_boundary = make_float3(current_voxel + step) * voxel_size;
        if (step.x < 0) next_boundary.x += voxel_size;
        if (step.y < 0) next_boundary.y += voxel_size;
        if (step.z < 0) next_boundary.z += voxel_size;

        float3 tMax = safe_divide(next_boundary-rays_o_local, rays_d);
        float3 tDelta = safe_divide(make_float3(step)*voxel_size, rays_d);


        while(true)
        {
            int vidx = get_index(num_voxel, num_voxel, num_voxel, &current_voxel.x);

            if (node_flags[vidx] == 1)
            {
                float3 voxel_center = min_corner + make_float3(current_voxel) * voxel_size + 0.5f * voxel_size;

                float2 bound = RayAABBIntersection(rays_o, rays_d, voxel_center, 0.5f * voxel_size);
                // assert(bound.x != -1 && bound.y != -1);
                total_length = total_length + (bound.y - bound.x);
            }

            if (tMax.x < tMax.y){
                if (tMax.x < tMax.z)
                {
                    current_voxel.x += step.x;
                    tMax.x += tDelta.x;
                }else{
                    current_voxel.z += step.z;
                    tMax.z += tDelta.z;
                }
            }else{
                if (tMax.y < tMax.z)
                {
                    current_voxel.y += step.y;
                    tMax.y += tDelta.y;
                }else{
                    current_voxel.z += step.z;
                    tMax.z += tDelta.z;
                }
            }

            if (current_voxel.x < 0 || 
                current_voxel.y < 0 || 
                current_voxel.z < 0 || 
                current_voxel.x >= num_voxel || 
                current_voxel.y >= num_voxel || 
                current_voxel.z >= num_voxel) break;
        }

        // if (cur_task_idx == 4870){
        //     printf("total length %f\n",total_length);
        // }
        if (total_length == 0) continue;

        current_voxel = make_int3(rays_o_local / voxel_size);
        current_voxel = clamp(current_voxel, 0, num_voxel-1);
        tMax = safe_divide(next_boundary-rays_o_local, rays_d);

        int count_sample = 0;

        bool accumulate = false;

        bool first_hit = false;

        int boundary_smaples = 4;
        int INSamples = Nsamples - 2 * boundary_smaples;

        while(true)
        {
            int vidx = get_index(num_voxel, num_voxel, num_voxel, &current_voxel.x);

            if (node_flags[vidx] == 1)
            {
                float3 voxel_center = min_corner + make_float3(current_voxel) * voxel_size + 0.5f * voxel_size;

                float2 bound = RayAABBIntersection(rays_o, rays_d, voxel_center, 0.5f * voxel_size);

                if (bound.x != -1 && bound.y != -1)
                {
                    if (accumulate) far = bound.y;
                    else{
                        near = bound.x;
                        far = bound.y;
                        accumulate = true;
                        if (first_hit == false)
                        {
                            first_hit = true;
                            uniform_sample_bound_v3(cur_z_vals+count_sample, cur_dists+count_sample,
                                                    near-voxel_size, near - 0.000001f, boundary_smaples);
                            count_sample = count_sample + boundary_smaples;
                        }
                    }
                }
                // assert(bound.x != -1 && bound.y != -1);
            }
            else if (accumulate)
            {
                int num_sample = (int)ceil(INSamples * (far - near) / total_length);
                num_sample = min(Nsamples - count_sample, num_sample);
                // uniform_sample_bound_v2(cur_z_vals+count_sample, near, far, num_sample);
                uniform_sample_bound_v3(cur_z_vals+count_sample, cur_dists+count_sample,
                                        near, far, num_sample);
                count_sample = count_sample + num_sample;
                accumulate = false;
            }

            if (tMax.x < tMax.y){
                if (tMax.x < tMax.z)
                {
                    current_voxel.x += step.x;
                    tMax.x += tDelta.x;
                }else{
                    current_voxel.z += step.z;
                    tMax.z += tDelta.z;
                }
            }else{
                if (tMax.y < tMax.z)
                {
                    current_voxel.y += step.y;
                    tMax.y += tDelta.y;
                }else{
                    current_voxel.z += step.z;
                    tMax.z += tDelta.z;
                }
            }

            if (current_voxel.x < 0 || 
                current_voxel.y < 0 || 
                current_voxel.z < 0 || 
                current_voxel.x >= num_voxel || 
                current_voxel.y >= num_voxel || 
                current_voxel.z >= num_voxel) break;
        }

        if (accumulate)
        {
            int num_sample = (int)ceil(INSamples * (far - near) / total_length);
            num_sample = min(Nsamples - count_sample, num_sample);
            // uniform_sample_bound_v2(cur_z_vals+count_sample, near, far, num_sample);
            uniform_sample_bound_v3(cur_z_vals+count_sample, cur_dists+count_sample,
                                     near, far, num_sample);
            count_sample = count_sample + num_sample;
        }

        boundary_smaples = min(Nsamples - count_sample, boundary_smaples);
        uniform_sample_bound_v3(cur_z_vals+count_sample, cur_dists+count_sample,
                                far + 0.000001f, far + voxel_size, boundary_smaples);

        // if (cur_task_idx == 0)
        // {
        //     for (int j=1; j<Nsamples; j++)
        //     {
        //         printf("%f %f %f\n", cur_z_vals[j-1], cur_z_vals[j], cur_z_vals[j] - cur_z_vals[j-1]);
        //     }
        // }

    }
}


void Sample_sparse_cuda(
    const at::Tensor rays_start,
    const at::Tensor rays_dir,
    const at::Tensor node_flags,
    const float cx,
    const float cy,
    const float cz,
    const float tile_size,
    const int num_voxel,
    at::Tensor &z_vals,
    at::Tensor &dists)
{
    int batchSize = rays_start.size(0);
    int Nsamples = z_vals.size(1);
    float3 center = make_float3(cx,cy,cz);

    unsigned int n_threads, n_blocks;

    n_threads = 512;
    n_blocks = min(65535, (batchSize + n_threads - 1) / n_threads);

    int add_num = batchSize % (n_blocks * n_threads);
    int base_jobs = batchSize / (n_blocks * n_threads);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    Sample_sparse_cuda_kernel<<<n_blocks, n_threads, 0, stream>>>(
        rays_start.contiguous().data_ptr<float>(),
        rays_dir.contiguous().data_ptr<float>(),
        node_flags.contiguous().data_ptr<short>(),
        center, tile_size, Nsamples, num_voxel,
        z_vals.contiguous().data_ptr<float>(),
        dists.contiguous().data_ptr<float>(),
        base_jobs, add_num);

    AT_CUDA_CHECK(cudaGetLastError());
    return;  
}

__global__ 
void Sample_uniform_cuda_kernel(
    const float* rays_start,
    const float* rays_dir,
    const float3 center,
    const float tile_size,
    const int Nsamples,
    float* z_vals, // B x Nsamples  z_vals
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

    float half_size = tile_size / 2.0f;

    for (int i=0; i<task_num; i++)
    {
        int cur_task_idx = start_task_idx + i;
        float* cur_z_vals = z_vals + cur_task_idx * Nsamples;

        float3 rays_o = make_float3(rays_start[cur_task_idx*3],
                                    rays_start[cur_task_idx*3+1],
                                    rays_start[cur_task_idx*3+2]);
        float3 rays_d = make_float3(rays_dir[cur_task_idx*3],
                                    rays_dir[cur_task_idx*3+1],
                                    rays_dir[cur_task_idx*3+2]);
        float2 near_far = RayAABBIntersection(rays_o, rays_d, center, half_size);

        float near = near_far.x;
        float far = near_far.y;

        uniform_sample_bound(cur_z_vals, near, far, Nsamples);
    }

}


__global__ 
void Sample_bg_cuda_kernel(
    const float* rays_start, // B x 3
    const float* rays_dir, //  B x 3
    const float* bg_zval, // B x 1
    const int Nsamples, 
    const float sample_range, 
    // const float* last_zval, // zval of last sample in tile 
    const float* tile_center,
    const float tile_size, 
    // const float voxel_size,
    float* z_vals, // B x Nsample 
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

    float3 center = make_float3(tile_center[0], tile_center[1], tile_center[2]);

    for (int i=0; i<task_num; i++)
    {
        int cur_task_idx = start_task_idx + i;

        float* cur_z_vals = z_vals + cur_task_idx * Nsamples;

        float bgdepth = bg_zval[cur_task_idx];
        // float lastdepth = last_zval[cur_task_idx];

        if (bgdepth == -1.0f) continue;

        float half_range = sample_range / 2.0f;
    
        float3 rays_o = make_float3(rays_start[cur_task_idx*3],
                                    rays_start[cur_task_idx*3+1],
                                    rays_start[cur_task_idx*3+2]);

        float3 rays_d = make_float3(rays_dir[cur_task_idx*3],
                                    rays_dir[cur_task_idx*3+1],
                                    rays_dir[cur_task_idx*3+2]);
        
        float2 bound = RayAABBIntersection(rays_o, rays_d, center, tile_size/2.0f);

        float near = bgdepth - half_range;
        float far = bgdepth + half_range;

        if (near <= bound.y)
        {
            near = bound.y + 0.00001f;
            far = near + sample_range;
        }

        uniform_sample_bound(cur_z_vals, near, far, Nsamples);
    }
}

void Sample_uniform_cuda(
    const at::Tensor rays_start,
    const at::Tensor rays_dir,
    const float cx,
    const float cy,
    const float cz,
    const float tile_size,
    at::Tensor &z_vals)
{
    int batchSize = rays_start.size(0);
    int Nsamples = z_vals.size(1);
    float3 center = make_float3(cx,cy,cz);
    
    unsigned int n_threads, n_blocks;

    n_threads = 512;
    n_blocks = min(65535, (batchSize + n_threads - 1) / n_threads);

    int add_num = batchSize % (n_blocks * n_threads);
    int base_jobs = batchSize / (n_blocks * n_threads);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    Sample_uniform_cuda_kernel<<<n_blocks, n_threads, 0, stream>>>(
        rays_start.contiguous().data_ptr<float>(),
        rays_dir.contiguous().data_ptr<float>(),
        center, tile_size, Nsamples,
        z_vals.contiguous().data_ptr<float>(),
        base_jobs, add_num);


    AT_CUDA_CHECK(cudaGetLastError());
    return;  
}


void Sample_bg_cuda(
    const at::Tensor rays_start,
    const at::Tensor rays_dir,
    const at::Tensor bg_zval,
    // const at::Tensor last_zval,
    const at::Tensor tile_center,
    const float tile_size,
    // const float voxel_size,
    const float sample_range, 
    at::Tensor &z_vals)
{
    int batchSize = rays_start.size(0);
    int Nsamples = z_vals.size(1);

    unsigned int n_threads, n_blocks;

    n_threads = 512;
    n_blocks = min(65535, (batchSize + n_threads - 1) / n_threads);

    int add_num = batchSize % (n_blocks * n_threads);
    int base_jobs = batchSize / (n_blocks * n_threads);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    Sample_bg_cuda_kernel<<<n_blocks, n_threads, 0, stream>>>(
        rays_start.contiguous().data_ptr<float>(),
        rays_dir.contiguous().data_ptr<float>(),
        bg_zval.contiguous().data_ptr<float>(),
        Nsamples, sample_range, 
        tile_center.contiguous().data_ptr<float>(), tile_size,
        z_vals.contiguous().data_ptr<float>(),
        base_jobs, add_num);


    AT_CUDA_CHECK(cudaGetLastError());
    return;  
}
