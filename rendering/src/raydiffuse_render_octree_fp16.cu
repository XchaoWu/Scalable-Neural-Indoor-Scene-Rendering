#include "raydiffuse_render_octree.h"
#include <c10/core/ScalarType.h>
#include <c10/util/irange.h>
#include "half_ope.h"

constexpr int num_block = 4; 

__device__ __constant__ float3 c_origin;

inline __device__
void volume_rendering(half3 &color, __half &transparency, half3 rgb, __half sigma, __half step)
{
    half alpha = 1.0f - hexp(-1.0f * sigma * step);
    half weight = alpha * transparency;
    color = color + weight * rgb;
    transparency = transparency * (1.0f - alpha);
}


__forceinline__ __device__ 
void block_traversal_rendering(
     half3 &color_diffuse, half &transparency,
     const float3 ray_o, const float3 ray_d,
     const half4* data_voxels, 
     const int* block_IndexMap, 
     const short* nodes_IndexMap, 
     const bool* nodes_sampleFlag, 
     const long* voxels_start, 
     int dilate_num_voxel_per_block,
     float sample_step,
     float block_size, float voxel_size,
     bool &first_hit, half trans_th,
     float* inverse_near, 
     short* netIdxs, 
     int cur_task_idx, int groupIdx,
     float tile_near, int tileIdx)
{

    int total_voxel_per_block = dilate_num_voxel_per_block*dilate_num_voxel_per_block*dilate_num_voxel_per_block;

    int3 current_block = make_int3(ray_o / block_size);
    int3 step = signf(ray_d);
    float3 next_boundary = make_float3(current_block + step) * block_size;
    if (step.x < 0) next_boundary.x += block_size;
    if (step.y < 0) next_boundary.y += block_size;
    if (step.z < 0) next_boundary.z += block_size;
    float3 tMax = safe_divide(next_boundary - ray_o, ray_d);
    float3 tDelta = safe_divide(make_float3(block_size * step.x, 
                                            block_size * step.y,
                                            block_size * step.z), ray_d);


    float near = 0, far = min_value(tMax);
    while (true)
    {

        int bidx = current_block.x * num_block * num_block + 
                   current_block.y * num_block + current_block.z;
        
        int nidx = block_IndexMap[bidx];

        if (nidx != -1)
        {
            const short* cur_nodes = nodes_IndexMap + nidx * total_voxel_per_block;
            const bool* cur_sampleFlag = nodes_sampleFlag + nidx *  total_voxel_per_block;
            long start = voxels_start[nidx];
            const half4* cur_data_voxels = data_voxels + start;

            float3 block_dilate_corner = make_float3(current_block) * block_size - voxel_size;

            float zval = near;

            while (zval < far)
            {
                float3 pts = ray_o + zval * ray_d;  
                int3 voxeloc = make_int3((pts - block_dilate_corner) / voxel_size);

                int vidx = voxeloc.x * (dilate_num_voxel_per_block * dilate_num_voxel_per_block) + 
                           voxeloc.y * dilate_num_voxel_per_block + voxeloc.z;
                if (!cur_sampleFlag[vidx])
                {
                    zval += sample_step;
                    continue;
                }
                if (first_hit == false && __hlt(transparency, trans_th))
                {
                    first_hit = true;
                    inverse_near[cur_task_idx] = __fdiv_rn(1.0f, zval+tile_near);
                    netIdxs[cur_task_idx] = groupIdx;
                }

                float3 fidx = (pts - block_dilate_corner) / voxel_size - 0.5f;

                half4 rgba = trilinear_base(cur_data_voxels, cur_nodes, 
                                             dilate_num_voxel_per_block,
                                             dilate_num_voxel_per_block,
                                             dilate_num_voxel_per_block, fidx);
                __half sigma = rgba.w;
                half3 diffuse = make_half3(rgba.x, rgba.y, rgba.z);

                float interval = min(sample_step, far - zval);

                volume_rendering(color_diffuse, transparency, diffuse, sigma, __float2half(interval));
                zval += sample_step;
            }
        }

        if (tMax.x < tMax.y){
            if (tMax.x < tMax.z)
            {
                current_block.x += step.x;
                tMax.x += tDelta.x;
            }else{
                current_block.z += step.z;
                tMax.z += tDelta.z;
            }
        }else{
            if (tMax.y < tMax.z)
            {
                current_block.y += step.y;
                tMax.y += tDelta.y;
            }else{
                current_block.z += step.z;
                tMax.z += tDelta.z;
            }
        }


        near = far;
        far = min_value(tMax);

        if (current_block.x < 0 || current_block.y < 0 || current_block.z < 0 || 
            current_block.x >= num_block || current_block.y >= num_block || current_block.z >= num_block)
        {
            break;
        }  

    }
}

__global__ 
void rendering_diffuse_octree_fp16_kernel(
    const float* rays_d,
    const int* visitedTiles,
    const int* groupMap, 
    const half4* data_voxels, 
    const int* block_IndexMap, 
    const short* nodes_IndexMap, 
    const bool* nodes_sampleFlag,
    const long* voxels_start, 
    const float* centers, 
    float tile_size, float voxel_size,
    float sample_step,
    int max_tracingTile, int num_voxel, 
    float* frame_diffuse,
    float* inverse_near, 
    short* netIdxs, 
    bool soft_hit, 
    half trans_th,
    float early_stop,
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

    float block_size = tile_size / num_block;

    while (cur_task_idx < end_task_idx)
    {
        const int* cur_visitedTiles = visitedTiles + cur_task_idx * max_tracingTile;


        half3 color_diffuse = make_half3(0.0f,0.0f,0.0f);
        __half transparency = __float2half(1.0f);

        float3 direction = make_float3(rays_d + cur_task_idx*3);

        bool first_hit = false;
        int groupIdx = -1;

        float2 bound;

        for (int i=0; i<max_tracingTile; i++)
        {

            int tileIdx = cur_visitedTiles[i];

            if (tileIdx == -1) break;

            if ( __hlt( transparency, __float2half(early_stop) ) ) break;


            groupIdx = groupMap[tileIdx];
            float3 tile_center = make_float3(centers + tileIdx*3);

            bound = RayAABBIntersection(c_origin, direction, tile_center, tile_size/2.0f);

            if (bound.x == -1) continue;

            float3 tile_corner = tile_center - tile_size / 2.0f;            


            float3 rays_o_local = c_origin + bound.x * direction - tile_corner;
            rays_o_local = clamp(rays_o_local, 0, tile_size - 0.00001f);

            block_traversal_rendering(color_diffuse, transparency,
                                    rays_o_local, direction, data_voxels,
                                    block_IndexMap + tileIdx * num_block * num_block * num_block, 
                                    nodes_IndexMap, nodes_sampleFlag,
                                    voxels_start, num_voxel, sample_step, block_size, voxel_size,
                                    first_hit, trans_th, inverse_near, netIdxs, cur_task_idx, groupIdx, bound.x, tileIdx);
        }

        if (soft_hit && first_hit == false && groupIdx != -1)
        {
            first_hit = true;
            inverse_near[cur_task_idx] = __fdiv_rn(1.0f, bound.y);
            netIdxs[cur_task_idx] = groupIdx;
        }

        frame_diffuse[cur_task_idx * 3] = __half2float(color_diffuse.x);
        frame_diffuse[cur_task_idx * 3 + 1] = __half2float(color_diffuse.y);
        frame_diffuse[cur_task_idx * 3 + 2] = __half2float(color_diffuse.z);
        cur_task_idx++;
    }
}


__host__
void rendering_diffuse_octree_fp16(
    at::Tensor rays_d, at::Tensor visitedTiles,
    at::Tensor origin,
    at::Tensor groupMap, 
    at::Tensor data_voxels,
    at::Tensor block_IndexMap, 
    at::Tensor nodes_IndexMap,
    at::Tensor nodes_sampleFlag,
    at::Tensor voxels_start,
    at::Tensor centers,
    float tile_size, float voxel_size,
    float sample_step, int num_voxel, int num_thread,
    float trans_th, float early_stop, bool soft_hit, 
    at::Tensor &frame_diffuse,
    at::Tensor &inverse_near, at::Tensor &netIdxs)
{

    assert(num_voxel == 18);

    int max_tracingTile = visitedTiles.size(2);
    int H = rays_d.size(0);
    int W = rays_d.size(1);
    int numPixel = H * W;
    int nblocks = min(65535, (numPixel + num_thread - 1) / num_thread);
    int base_num = numPixel / (nblocks * num_thread);
    int extra_num = numPixel - base_num * (nblocks * num_thread);

    cudaMemcpyToSymbol( c_origin.x, origin.contiguous().data_ptr<float>(), sizeof(float)*3, 0, cudaMemcpyDeviceToDevice);

    rendering_diffuse_octree_fp16_kernel<<<nblocks, num_thread>>>(
        rays_d.contiguous().data_ptr<float>(), 
        visitedTiles.contiguous().data_ptr<int>(), 
        groupMap.contiguous().data_ptr<int>(), 
        (half4*)data_voxels.contiguous().data_ptr<at::Half>(), 
        block_IndexMap.contiguous().data_ptr<int>(), 
        nodes_IndexMap.contiguous().data_ptr<short>(),
        nodes_sampleFlag.contiguous().data_ptr<bool>(), 
        voxels_start.contiguous().data_ptr<long>(), 
        centers.contiguous().data_ptr<float>(), 
        tile_size, voxel_size, sample_step, max_tracingTile, num_voxel, 
        frame_diffuse.contiguous().data_ptr<float>(), 
        inverse_near.contiguous().data_ptr<float>(), 
        netIdxs.contiguous().data_ptr<short>(), soft_hit,
        __float2half(trans_th), early_stop, extra_num, base_num);

    AT_CUDA_CHECK(cudaGetLastError());

    return;

}