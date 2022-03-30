#include "raytile_tracing.h"


__device__ __constant__ float3 c_origin;
__device__ __constant__ float3 c_scene_min_corner;
__device__ __constant__ int3 c_tile_shape;
__device__ __constant__ float3  c_scene_size;

__global__
void tracing_tiles_kernel(
    const float* rays_dir, // B x 3 
    const int* IndexMap, // dense tile Idx -> sparce tile Idx 
    const float tile_size, 
    const int max_tracingTile, 
    int* visitedTiles, // B x max_tracingTile if -1 break 
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
        int* cur_visitedTiles = visitedTiles + cur_task_idx * max_tracingTile;

        for (int j=0; j<max_tracingTile; j++)
        {
            cur_visitedTiles[j] = -1;
        }

        float3 rays_d = make_float3(rays_dir + cur_task_idx*3);

        float3 rays_o_local = c_origin - c_scene_min_corner;

        if (rays_o_local.x < 0 || rays_o_local.x >= c_scene_size.x ||
            rays_o_local.y < 0 || rays_o_local.y >= c_scene_size.y ||
            rays_o_local.z < 0 || rays_o_local.z >= c_scene_size.z )
        {
            float2 bound = RayAABBIntersection(c_origin, rays_d, 
                                               c_scene_min_corner + c_scene_size / 2.0f, 
                                               c_scene_size / 2.0f);
            rays_o_local = rays_o_local + bound.x * rays_d;
            rays_o_local = clamp(rays_o_local, 0, c_scene_size-0.000001f);
        }

        voxel_traversal(&rays_o_local.x, &rays_d.x, IndexMap,
                        cur_visitedTiles, max_tracingTile,
                        c_tile_shape.x, c_tile_shape.y, c_tile_shape.z, tile_size);
        cur_task_idx++;
    }
}


__host__ 
void tracing_tiles(
    at::Tensor rays_d, at::Tensor IndexMap, 
    at::Tensor origin, at::Tensor tile_shape,
    at::Tensor scene_min_corner, at::Tensor scene_size,
    float tile_size, int num_thread,
    at::Tensor &visitedTiles)
{
    int max_tracingTile = visitedTiles.size(2);
    int H = rays_d.size(0);
    int W = rays_d.size(1);
    int numPixel = H * W;
    int num_block = min(65535, (numPixel + num_thread - 1) / num_thread);
    int base_num = numPixel / (num_block * num_thread);
    int extra_num = numPixel - base_num * (num_block * num_thread);

    cudaMemcpyToSymbol( c_origin.x, origin.contiguous().data_ptr<float>(), sizeof(float)*3, 0, cudaMemcpyDeviceToDevice);
    cudaMemcpyToSymbol( c_scene_min_corner.x, scene_min_corner.contiguous().data_ptr<float>(), sizeof(float)*3, 0, cudaMemcpyDeviceToDevice);
    cudaMemcpyToSymbol( c_tile_shape.x, tile_shape.contiguous().data_ptr<int>(), sizeof(int)*3, 0, cudaMemcpyDeviceToDevice);
    cudaMemcpyToSymbol( c_scene_size.x, scene_size.contiguous().data_ptr<float>(), sizeof(float)*3, 0, cudaMemcpyDeviceToDevice);

    tracing_tiles_kernel<<<num_block, num_thread>>>(
        rays_d.contiguous().data_ptr<float>(), 
        IndexMap.contiguous().data_ptr<int>(), 
        tile_size, max_tracingTile, 
        visitedTiles.contiguous().data_ptr<int>(), 
        extra_num, base_num);

    AT_CUDA_CHECK(cudaGetLastError());

    return;

}

