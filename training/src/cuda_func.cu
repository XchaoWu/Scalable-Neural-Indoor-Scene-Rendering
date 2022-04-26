#include "cuda_func.h"
#include "interpolation.h"
#include <cnpy.h>
// template<typename T>
// __global__ void initKernel(T * devPtr, const T val, const size_t nwords)
// {
//     int tidx = threadIdx.x + blockDim.x * blockIdx.x;
//     int stride = blockDim.x * gridDim.x;

//     for(; tidx < nwords; tidx += stride)
//         devPtr[tidx] = val;
// }


__global__ 
void computing_rays_kernel(
    const int height,
    const int width,
    const float* K,
    const float* C2W, 
    float3* rays_o,
    float3* rays_d,
    int add_num, int base_jobs)
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


    for (int i=0; i<task_num; i++)
    {

        int cur_task_idx = start_task_idx + i;

        int _y = cur_task_idx / width;
        int _x = cur_task_idx % width;

        float3 ray_o, ray_d;
        get_rays(_x, _y, K, C2W, ray_o, ray_d);
        rays_o[cur_task_idx] = ray_o;
        rays_d[cur_task_idx] = ray_d;
    }
}

__global__ 
void tracingTile_kernel(
    const int height,
    const int width,
    const float3* rays_start,
    const float3* rays_dir, 
    const int* IndexMap,
    const int3 tile_shape,
    const float tile_size,
    const int max_tracingTile,
    int* visitedTiles,
    const float3 scene_min_corner,
    int add_num, int base_jobs)
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

    float3 zeros = make_float3(0.0f, 0.0f, 0.0f);
    float3 scene_size = make_float3(tile_shape) * tile_size;

    // if (cur_task_idx == 735772)
    // {
    //     printf("scene_size: %f %f %f\n",
    //     scene_size.x,scene_size.y,scene_size.z);
    // }

    for (int i=0; i<task_num; i++)
    {
        int cur_task_idx = start_task_idx + i;

        int* cur_visitedTiles = visitedTiles+cur_task_idx*max_tracingTile;

        for (int j=0; j<max_tracingTile; j++)
        {
            cur_visitedTiles[j] = -1;
        }

        float3 rays_o = rays_start[cur_task_idx];
        float3 rays_d = rays_dir[cur_task_idx];
        rays_d = normalize(rays_d);

        float3 ray_o_local = rays_o - scene_min_corner;

        if (ray_o_local.x < 0 || ray_o_local.x >= scene_size.x ||
            ray_o_local.y < 0 || ray_o_local.y >= scene_size.y ||
            ray_o_local.z < 0 || ray_o_local.z >= scene_size.z )
        {
            float2 bound = RayAABBIntersection(rays_o, rays_d, 
                                scene_min_corner + scene_size / 2.0f, scene_size / 2.0f);
            ray_o_local = ray_o_local + bound.x * rays_d;
        }

        ray_o_local = clamp(ray_o_local, zeros, scene_size - 0.000001f);
        voxel_traversal(&ray_o_local.x, &rays_d.x, IndexMap, 
                        cur_visitedTiles, max_tracingTile, 
                        tile_shape.x, tile_shape.y, tile_shape.z, tile_size);

        // if (cur_task_idx == 735772)
        // {
        //     printf("ray_o: %f %f %f\n", rays_o.x, rays_o.y, rays_o.z);
        //     printf("ray_local: %f %f %f\n", ray_o_local.x, ray_o_local.y, ray_o_local.z);
        //     for (int j=0; j<max_tracingTile; j++)
        //     {
        //         printf("visited %d\n", cur_visitedTiles[j]);
        //     }   
        // }
    }
}


__global__ 
void ray_firstHit_inTile_kernel(
    const int height,
    const int width,
    const float3* rays_start,
    const float3* rays_dir, 
    const float3* vertices,
    const int3* faces,
    const float3* centers,
    const int* BConFaceIdx, // voxel 
    const int2* BConFaceNum, //  voxel numTile x numVoxel
    const float tile_size,
    const int max_tracingTile,
    const int num_block,
    const int* visitedTiles, // B x max_tracing
    float* firstHits, // B x max_tracing
    int* firstHitsFace, 
    int add_num, int base_jobs)
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

    int item = width * max_tracingTile;
    // int voxels_per_tile = num_voxel * num_voxel * num_voxel;
    int blocks_per_tile = num_block * num_block * num_block;
    float block_size = tile_size / num_block;

    for (int i=0; i<task_num; i++)
    {
        int cur_task_idx = start_task_idx + i;

        firstHitsFace[cur_task_idx] = -1;

        int _y = cur_task_idx / item;
        int _x = (cur_task_idx - _y * item) / max_tracingTile;
        int ray_idx = _y * width + _x;
        int tileIdx = visitedTiles[cur_task_idx];

        if (tileIdx == -1) continue;

        const int2* cur_BConFaceNum = BConFaceNum + tileIdx * blocks_per_tile;

        float3 rays_o = rays_start[ray_idx];
        float3 rays_d = rays_dir[ray_idx];
        // float3 rays_o, rays_d;
        // get_rays(_x, _y, K, C2W, rays_o, rays_d);
        // rays_d = normalize(rays_d);

        float3 tcenter = centers[tileIdx];
        float3 min_corner = tcenter - tile_size / 2.0f;

        float2 bound = RayAABBIntersection(rays_o, rays_d, tcenter, tile_size/2.0f);

        float3 new_rays_o = rays_o + bound.x * rays_d - min_corner;
        new_rays_o = clamp(new_rays_o, 0.0f, tile_size-0.000001f);

        // ===== ray voxel dense tracing ========
        int3 current_block = make_int3(new_rays_o / block_size);
        int3 step = signf(rays_d); 
        float3 next_boundary = make_float3(current_block + step) * block_size;
        if (step.x < 0) next_boundary.x += block_size;
        if (step.y < 0) next_boundary.y += block_size;
        if (step.z < 0) next_boundary.z += block_size;
        float3 tMax = safe_divide(next_boundary-new_rays_o, rays_d);
        float3 tDelta = safe_divide(make_float3(step)*block_size, rays_d);

        while(true)
        {
            int bidx = current_block.z + current_block.y * num_block + current_block.x * num_block * num_block;
            float3 block_center = min_corner + make_float3(current_block) * block_size  + block_size / 2.0f;
            int start_face = cur_BConFaceNum[bidx].x;
            int num_face = cur_BConFaceNum[bidx].y;
            const int* face_idx = BConFaceIdx + start_face;
            float near = INF;
            int faceindex = -1;
            for (int j=0; j<num_face; j++)
            {
                int3 vidx = faces[face_idx[j]];
                float3 v0 = vertices[vidx.x];
                float3 v1 = vertices[vidx.y];
                float3 v2 = vertices[vidx.z];

                float t;
                float3 v0v1 = v1 - v0;
                float3 v1v2 = v2 - v1;
                float3 tri_normal = normalize(cross(v0v1, v1v2));
                float sim = dot(rays_d, tri_normal);

                if (sim > 0)
                {
                    t = RayTriangleIntersection(rays_o, rays_d, v0, v2, v1, 0.0f);
                }else{
                    t = RayTriangleIntersection(rays_o, rays_d, v0, v1, v2, 0.0f);
                }

                if (t > 0 && t < near)
                {
                    float3 diff = fabs(rays_o + t * rays_d - block_center) - block_size/2.0f;
                    if (diff.x<=THRESH && diff.y<=THRESH && diff.z<=THRESH)
                    {
                        near = t;
                        faceindex = face_idx[j];
                    }
                }
            }
            
            if (near != INF)
            {
                firstHits[cur_task_idx] = near;
                firstHitsFace[cur_task_idx] = faceindex;
                break;
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

            if (current_block.x < 0 || 
                current_block.y < 0 || 
                current_block.z < 0 || 
                current_block.x >= num_block || 
                current_block.y >= num_block || 
                current_block.z >= num_block) break;

        }

    }

}

__global__ 
void ray_doubleHit_inTile_kernel(
    const int height,
    const int width,
    const float3* rays_start,
    const float3* rays_dir, 
    const float3* vertices,
    const int3* faces,
    const float3* centers,
    const int* BConFaceIdx, // voxel 
    const int2* BConFaceNum, //  voxel numTile x numVoxel
    const float tile_size,
    const int max_tracingTile,
    const int num_block,
    const int* visitedTiles, // B x max_tracing
    float2* doubleHits, // B x max_tracing
    int add_num, int base_jobs)
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

    int item = width * max_tracingTile;
    // int voxels_per_tile = num_voxel * num_voxel * num_voxel;
    int blocks_per_tile = num_block * num_block * num_block;
    float block_size = tile_size / num_block;

    for (int i=0; i<task_num; i++)
    { 
        int cur_task_idx = start_task_idx + i;

        doubleHits[cur_task_idx].x = -1;
        doubleHits[cur_task_idx].y = -1;

        int _y = cur_task_idx / item;
        int _x = (cur_task_idx - _y * item) / max_tracingTile;
        int ray_idx = _y * width + _x;
        int tileIdx = visitedTiles[cur_task_idx];

        if (tileIdx == -1) continue;

        const int2* cur_BConFaceNum = BConFaceNum + tileIdx * blocks_per_tile;

        float3 rays_o = rays_start[ray_idx];
        float3 rays_d = rays_dir[ray_idx];
        // float3 rays_o, rays_d;
        // get_rays(_x, _y, K, C2W, rays_o, rays_d);
        // rays_d = normalize(rays_d);

        float3 tcenter = centers[tileIdx];
        float3 min_corner = tcenter - tile_size / 2.0f;

        float2 bound = RayAABBIntersection(rays_o, rays_d, tcenter, tile_size/2.0f);

        float3 new_rays_o = rays_o + bound.x * rays_d - min_corner;
        new_rays_o = clamp(new_rays_o, 0.0f, tile_size-0.000001f);

        // ===== ray voxel dense tracing ========
        int3 current_block = make_int3(new_rays_o / block_size);
        int3 step = signf(rays_d); 
        float3 next_boundary = make_float3(current_block + step) * block_size;
        if (step.x < 0) next_boundary.x += block_size;
        if (step.y < 0) next_boundary.y += block_size;
        if (step.z < 0) next_boundary.z += block_size;
        float3 tMax = safe_divide(next_boundary-new_rays_o, rays_d);
        float3 tDelta = safe_divide(make_float3(step)*block_size, rays_d);


        float near = INF; float second_near = INF;
        while(true)
        {
            int bidx = current_block.z + current_block.y * num_block + current_block.x * num_block * num_block;
            float3 voxel_center = min_corner + make_float3(current_block) * block_size  + block_size / 2.0f;
            int start_face = cur_BConFaceNum[bidx].x;
            int num_face = cur_BConFaceNum[bidx].y;
            const int* face_idx = BConFaceIdx + start_face;
            for (int j=0; j<num_face; j++)
            {
                int3 vidx = faces[face_idx[j]];
                float3 v0 = vertices[vidx.x];
                float3 v1 = vertices[vidx.y];
                float3 v2 = vertices[vidx.z];

                float t;
                float3 v0v1 = v1 - v0;
                float3 v1v2 = v2 - v1;
                float3 tri_normal = normalize(cross(v0v1, v1v2));
                float sim = dot(rays_d, tri_normal);

                if (sim > 0)
                {
                    t = RayTriangleIntersection(rays_o, rays_d, v0, v2, v1, 0.0f);
                }else{
                    t = RayTriangleIntersection(rays_o, rays_d, v0, v1, v2, 0.0f);
                }

                if (t > 0 && t < second_near)
                {
                    float3 diff = fabs(rays_o + t * rays_d - voxel_center) - block_size/2.0f;
                    if (diff.x<=THRESH && diff.y<=THRESH && diff.z<=THRESH)
                    {
                        if (t < near)
                        {
                            second_near = near;
                            near = t;
                        }else{
                            second_near = t;
                        }
                    }
                }
            }
            
            if (second_near != INF)
            {
                break;
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

            if (current_block.x < 0 || 
                current_block.y < 0 || 
                current_block.z < 0 || 
                current_block.x >= num_block || 
                current_block.y >= num_block || 
                current_block.z >= num_block) break;

        }
        if (near != INF)
        {
            doubleHits[cur_task_idx].x = near;
            if (second_near != INF) doubleHits[cur_task_idx].y = second_near;
        }

    }

}


__global__ 
void get_firstHit_kernel(
    const int height, 
    const int width,
    const float3* rays_start,
    const float3* rays_dir, 
    const float3* vertices,
    const int3* faces,
    const float3* colors,
    const int* visitedTiles,
    const float* firstHits, // B x max_tracing
    const int* firstHitFace,
    const int max_tracingTile,
    int* ref_tag,
    float3* bgcolors,
    int add_num, int base_jobs)
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

    for (int i=0; i<task_num; i++)
    { 
        int cur_task_idx = start_task_idx + i;

        const float* cur_firstHits = firstHits + cur_task_idx * max_tracingTile;
        const int* cur_firstHitFace = firstHitFace + cur_task_idx * max_tracingTile;
        int* cur_ref_tag = ref_tag + cur_task_idx * max_tracingTile;


        float3 rays_o = rays_start[cur_task_idx];
        float3 rays_d = rays_dir[cur_task_idx];


        for (int j=0; j<max_tracingTile; j++)
        {
            if (cur_firstHits[j] != 0)
            {
                float firsthit = cur_firstHits[j];
                int hitface = cur_firstHitFace[j];
                cur_ref_tag[j] = 1;

                // barycentric interpolation 
                float3 P = rays_o + firsthit * rays_d;
                int3 vidx = faces[hitface];
                float3 A = vertices[vidx.x];
                float3 B = vertices[vidx.y];
                float3 C = vertices[vidx.z];
                float3 CA = colors[vidx.x];
                float3 CB = colors[vidx.y];
                float3 CC = colors[vidx.z];
                float3 AB = B - A;
                float3 AC = C - A;
                float3 AP = P - A;
                float3 BC = C - B;
                float3 BP = P - B;
                float areaABC = norm(cross(AB, AC));
                float areaABP = norm(cross(AB, AP));
                float areaBCP = norm(cross(BC, BP));
                float areaACP = norm(cross(AP, AC));
                float u = areaBCP / areaABC;
                float v = areaACP / areaABC;
                float w = areaABP / areaABC;
                // printf("%f %f %f\n", u, v, w);
                bgcolors[cur_task_idx] = u * CA + v * CB + w * CC;

                break;
            }else{
                cur_ref_tag[j] = 1;
            }
        }
    }
}


__global__ 
void get_VisImg_kernel(
    const int imgIdx, 
    const int num_camera,
    const int* visitedTiles,
    const float* firstHits, // B x max_tracing
    const int max_tracingTile,
    int* VisImg, 
    int add_num, int base_jobs)
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

    for (int i=0; i<task_num; i++)
    {
        int cur_task_idx = start_task_idx + i;

        const float* cur_firstHits = firstHits + cur_task_idx * max_tracingTile;
        const int* cur_visitedTiles = visitedTiles + cur_task_idx * max_tracingTile;
        
        int flag = -1;

        // if (cur_task_idx == 735772)
        // {
        //     for (int j=0; j<max_tracingTile; j++)
        //     {
        //         printf("%f\n", cur_firstHits[j]);
        //     }
        // }
        
        for (int j=0; j<max_tracingTile; j++)
        {
            if (cur_firstHits[j] != 0)
            {
                flag = j;
                break;
            }
        }
        if (flag == -1) continue;

        for (int j=0; j<=flag; j++)
        {
            int tileIdx = cur_visitedTiles[j];
            VisImg[tileIdx*num_camera+imgIdx] = 1;
        }

    }
}


__global__ 
void get_trainData_kernel(
    const int trainTileIdx,
    const float3* rays_start,
    const float3* rays_dir, 
    const float3* vertices,
    const int3* faces,
    const int* visitedTiles,
    const float2* doubleHits,
    const int max_tracingTile,
    float* bgdepth,
    int* valid, 
    int add_num, int base_jobs)
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

    for (int i=0; i<task_num; i++)
    {
        int cur_task_idx = start_task_idx + i;

        // float3 rays_o = rays_start[cur_task_idx];
        // float3 rays_d = rays_dir[cur_task_idx];

        // const float* cur_firstHits = firstHits + cur_task_idx * max_tracingTile;
        // const int* cur_firstHitFace = firstHitFace + cur_task_idx * max_tracingTile;
        const float2* cur_doubleHits = doubleHits + cur_task_idx * max_tracingTile;
        const int* cur_visitedTiles = visitedTiles + cur_task_idx * max_tracingTile;

        float first_zval = -1, second_zval = -1;
        int flag_first = -1;

        for (int j=0; j<max_tracingTile; j++)
        {
            if (cur_doubleHits[j].x != -1){
                if (flag_first == -1) {
                    flag_first = j;
                    first_zval = cur_doubleHits[j].x;
                }else{
                    second_zval = cur_doubleHits[j].x;
                    break;
                }
                // if (cur_doubleHits[j].y != -1)
                // {
                //     second_zval = cur_doubleHits[j].x;
                //     break;
                // }
            }
        }


        for (int j=0; j<=flag_first; j++)
        {
            // printf("%d %d\n", cur_visitedTiles[j], trainTileIdx);
            if (cur_visitedTiles[j] == trainTileIdx)
            {
                if (j == flag_first)
                {
                    // if (second_zval == -1) second_zval = -1.0f;
                    bgdepth[cur_task_idx] = second_zval;
                }else{
                    bgdepth[cur_task_idx] = first_zval;
                }
                valid[cur_task_idx] = 1;
                break;
            }
        }
    }
}

__global__ 
void get_trainData_kernel_v2(
    const int trainTileIdx,
    const float3* rays_start,
    const float3* rays_dir, 
    const float3* vertices,
    const int3* faces,
    const float3* colors,
    const int* visitedTiles,
    const float2* doubleHits,
    float* check_first,
    // const float* firstHits, // B x max_tracing
    // const int* firstHitFace,
    const int max_tracingTile,
    // float3* bgcolors,
    float* bgdepth,
    int* valid,
    int add_num, int base_jobs)
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

    for (int i=0; i<task_num; i++)
    {
        int cur_task_idx = start_task_idx + i;

        // float3 rays_o = rays_start[cur_task_idx];
        // float3 rays_d = rays_dir[cur_task_idx];

        // const float* cur_firstHits = firstHits + cur_task_idx * max_tracingTile;
        // const int* cur_firstHitFace = firstHitFace + cur_task_idx * max_tracingTile;
        const float2* cur_doubleHits = doubleHits + cur_task_idx * max_tracingTile;
        const int* cur_visitedTiles = visitedTiles + cur_task_idx * max_tracingTile;

        float first_zval = -1, second_zval = -1;
        int flag_first = -1;


        // int ref_idx = -1;
        // for (int j=0; j<max_tracingTile; j++)
        // {
        //     if (cur_visitedTiles[j] == 2185)
        //     {
        //         ref_idx = j;
        //         break;
        //     }
        // }
        for (int j=0; j<max_tracingTile; j++)
        {
            if (cur_doubleHits[j].x != -1){
                if (flag_first == -1) {
                    flag_first = j;
                    first_zval = cur_doubleHits[j].x;
                }else{
                    second_zval = cur_doubleHits[j].x;
                    break;
                }
                // if (cur_doubleHits[j].y != -1)
                // {
                //     second_zval = cur_doubleHits[j].x;
                //     break;
                // }
            }
        }

        if (flag_first != -1)
            check_first[cur_task_idx] = first_zval;
        else check_first[cur_task_idx] = 0;

        // if (ref_idx != -1)
        // {
        // printf("ref %d  first %d\n", ref_idx, flag_first);
        // }

        for (int j=0; j<=flag_first; j++)
        {
            // printf("%d %d\n", cur_visitedTiles[j], trainTileIdx);
            if (cur_visitedTiles[j] == trainTileIdx)
            {
                if (j == flag_first)
                {
                    bgdepth[cur_task_idx] = second_zval;
                }else{
                    bgdepth[cur_task_idx] = first_zval;
                }
                valid[cur_task_idx] = 1;
                break;
            }
        }
    }
}

void get_trainData(
    const int trainTileIdx,
    const std::string img_path,
    const std::vector<float3> vertices,
    const std::vector<int3> faces, 
    const std::vector<float3> colors,
    const std::vector<float3> centers,
    const std::vector<int> IndexMap,
    const std::vector<int> BConFaceIdx,
    const std::vector<int2> BConFaceNum,
    const std::vector<int> imgIdxs,
    const float* Ks, const float* C2Ws,
    const int num_camera,
    const int3 tile_shape,
    const float3 scene_min_corner,
    const float tile_size,
    const int max_tracingTile,
    const int num_block,
    const int height, const int width,
    const bool debug,
    std::vector<float3> &data)
{
    int num_vertices = (int)vertices.size();
    int num_faces = (int)faces.size();
    int num_colors = (int)colors.size();
    int num_BConFaceIdx = (int)BConFaceIdx.size();
    int num_BConFaceNum = (int)BConFaceNum.size();
    int numPixels = height * width;
    int numTile = (int)centers.size();
    int total_tile = tile_shape.x * tile_shape.y * tile_shape.z;

    float3 *_vertices, *_centers, *_colors;
    int3 *_faces;
    int2 *_BConFaceNum;
    int *_BConFaceIdx;
    int *_IndexMap, *_visitedTiles, *_valid;
    float *_Ks, *_C2Ws, *_bgdepth;
    float2* _doubleHits;
    float3 *_rays_o, *_rays_d;

    cudaMalloc((void**)&_Ks, sizeof(float)*9*num_camera);
    cudaMemcpy(_Ks, Ks, sizeof(float)*9*num_camera, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&_C2Ws, sizeof(float)*12*num_camera);
    cudaMemcpy(_C2Ws, C2Ws, sizeof(float)*12*num_camera, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&_centers, sizeof(float3)*numTile);
    cudaMemcpy(_centers, centers.data(), sizeof(float3)*numTile, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&_IndexMap, sizeof(int)*total_tile);
    cudaMemcpy(_IndexMap, IndexMap.data(), sizeof(int)*total_tile, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&_vertices, sizeof(float3)*num_vertices);
    cudaMemcpy(_vertices, vertices.data(), sizeof(float3)*num_vertices, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&_faces, sizeof(int3)*num_faces);
    cudaMemcpy(_faces, faces.data(), sizeof(int3)*num_faces, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&_colors, sizeof(float3)*num_colors);
    cudaMemcpy(_colors, colors.data(), sizeof(float3)*num_colors, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&_BConFaceNum, sizeof(int2)*num_BConFaceNum);
    cudaMemcpy(_BConFaceNum, BConFaceNum.data(), sizeof(int2)*num_BConFaceNum, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&_BConFaceIdx, sizeof(int)*num_BConFaceIdx);
    cudaMemcpy(_BConFaceIdx, BConFaceIdx.data(), sizeof(int)*num_BConFaceIdx, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&_visitedTiles, sizeof(int)*max_tracingTile*numPixels);
    cudaMalloc((void**)&_doubleHits, sizeof(float2)*max_tracingTile*numPixels);
    // cudaMalloc((void**)&_firstHitFace, sizeof(int)*max_tracingTile*numPixels);
    cudaMalloc((void**)&_rays_o, sizeof(float3)*numPixels);
    cudaMalloc((void**)&_rays_d, sizeof(float3)*numPixels);
    // cudaMalloc((void**)&_bgcolors, sizeof(float3)*numPixels);
    cudaMalloc((void**)&_bgdepth, sizeof(float)*numPixels);
    cudaMalloc((void**)&_valid, sizeof(int)*numPixels);

    // float3 *bgcolors, *rays_o, *rays_d; 
    float3 *rays_o, *rays_d; 
    float* bgdepth;
    int* valid;
    // cudaHostAlloc( (void**)&bgcolors,sizeof(float3)*numPixels,cudaHostAllocDefault);
    cudaHostAlloc( (void**)&bgdepth,sizeof(float)*numPixels,cudaHostAllocDefault);
    cudaHostAlloc( (void**)&rays_o,sizeof(float3)*numPixels,cudaHostAllocDefault);
    cudaHostAlloc( (void**)&rays_d,sizeof(float3)*numPixels,cudaHostAllocDefault);
    cudaHostAlloc( (void**)&valid,sizeof(int)*numPixels,cudaHostAllocDefault);

    unsigned int n_threads, n_blocks, n_threads2, n_blocks2;
    n_threads = 512;
    n_blocks = min(65535, (numPixels + n_threads - 1) / n_threads);
    n_threads2 = 512;
    n_blocks2 = min(65535, (numPixels*max_tracingTile + n_threads2 - 1) / n_threads2);

    int add_num = numPixels % (n_blocks * n_threads);
    int base_jobs = numPixels / (n_blocks * n_threads);

    int add_num2 = (numPixels*max_tracingTile) % (n_blocks2 * n_threads2);
    int base_jobs2 = (numPixels*max_tracingTile) / (n_blocks2 * n_threads2);

    tqdm bar;
    for (int j=0; j<imgIdxs.size(); j++)
    {
        bar.progress(j, (int)imgIdxs.size());
        int i = imgIdxs[j];
        // cudaMemset(_bgcolors, 0, sizeof(float3)*numPixels);
        cudaMemset(_valid, 0, sizeof(int)*numPixels);

        // get rays
        computing_rays_kernel<<<n_blocks, n_threads>>>(
            height, width, _Ks + i * 9, _C2Ws + i * 12, _rays_o, _rays_d, add_num, base_jobs);

        // get the tileIdx for each ray 
        tracingTile_kernel<<<n_blocks, n_threads>>>(
            height, width, _rays_o, _rays_d, _IndexMap, tile_shape,
            tile_size, max_tracingTile, _visitedTiles, scene_min_corner, add_num, base_jobs);

        ray_doubleHit_inTile_kernel<<<n_blocks2, n_threads2>>>(
            height, width, _rays_o, _rays_d, _vertices, _faces, _centers,
            _BConFaceIdx, _BConFaceNum, tile_size, max_tracingTile, num_block,
            _visitedTiles, _doubleHits, add_num2, base_jobs2);
        
        get_trainData_kernel<<<n_blocks, n_threads>>>(
            trainTileIdx, _rays_o, _rays_d, _vertices, _faces, _visitedTiles,
            _doubleHits, max_tracingTile, _bgdepth, _valid, add_num, base_jobs);

        cudaMemcpy( bgdepth, _bgdepth, sizeof(float)*numPixels, cudaMemcpyDeviceToHost );
        cudaMemcpy( rays_o, _rays_o, sizeof(float3)*numPixels, cudaMemcpyDeviceToHost );
        cudaMemcpy( rays_d, _rays_d, sizeof(float3)*numPixels, cudaMemcpyDeviceToHost );
        cudaMemcpy( valid, _valid, sizeof(int)*numPixels, cudaMemcpyDeviceToHost );


        std::string path = img_path + "/" + std::to_string(i) + ".png";
        cv::Mat gt = cv::imread(path);
        cv::cvtColor(gt, gt, cv::COLOR_BGR2RGB);
        gt.convertTo(gt, CV_32FC3, 1.0f/255.0f);

        if (gt.isContinuous())
        {
            float3* gtcolors = (float3*)gt.data;
            for (int b = 0; b<numPixels; b++)
            {
                if (valid[b] == 0) continue;
                data.emplace_back(rays_o[b]);
                data.emplace_back(rays_d[b]);
                data.emplace_back(make_float3(bgdepth[b],bgdepth[b],bgdepth[b]));
                data.emplace_back(gtcolors[b]);
            }    
        }else{
            printf("img Not Contiguous!\n");
        }

    
        // cv::Mat img(height, width, CV_32FC3, (float*)bgcolors);
        // img.convertTo(img, CV_8UC3, 255.0f);
        // cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
        // std::string save_path = "bgcolor" + std::to_string(i) + ".png";
        // cv::imwrite(save_path, img);
        if (debug == true)
        {
            std::string save_path = "gtcolor" + std::to_string(i) + ".png";
            float3* gtcolors = (float3*)gt.data;
            for (int b = 0; b<numPixels; b++)
            {
                if (valid[b] == 0) gtcolors[b] *= 0.4;
            }
            gt.convertTo(gt, CV_8UC3, 255.0f);
            cv::cvtColor(gt, gt, cv::COLOR_RGB2BGR);
            cv::imwrite(save_path, gt);
        }
    }
    bar.finish();

    cudaFreeHost(bgdepth);
    cudaFreeHost(valid);
    cudaFreeHost(rays_o);
    cudaFreeHost(rays_d);
    cudaFree(_bgdepth);
    cudaFree(_valid);
    cudaFree(_rays_o);
    cudaFree(_rays_d);
    cudaFree(_doubleHits);
    // cudaFree(_firstHitFace);
    cudaFree(_Ks);
    cudaFree(_C2Ws);
    cudaFree(_vertices);
    cudaFree(_faces);
    cudaFree(_colors);
    cudaFree(_centers);
    cudaFree(_BConFaceNum);
    cudaFree(_BConFaceIdx);
    cudaFree(_IndexMap);
    cudaFree(_visitedTiles);

    cudaError_t err = cudaGetLastError();

    if ( err != cudaSuccess )
    {
       printf("CUDA Error: %s\n", cudaGetErrorString(err));       
    }
}

void get_trainData_v2(
    const int trainTileIdx,
    const std::string img_path,
    const std::string diffuse_path,
    const std::vector<float3> vertices,
    const std::vector<int3> faces, 
    const std::vector<float3> colors,
    const std::vector<float3> centers,
    const std::vector<int> IndexMap,
    const std::vector<int> BConFaceIdx,
    const std::vector<int2> BConFaceNum,
    const std::vector<int> imgIdxs,
    const float* Ks, const float* C2Ws,
    const int num_camera,
    const int3 tile_shape,
    const float3 scene_min_corner,
    const float tile_size,
    const int max_tracingTile,
    const int num_block,
    const int height, const int width,
    const bool debug,
    std::vector<float> &data)
{
    int num_vertices = (int)vertices.size();
    int num_faces = (int)faces.size();
    int num_colors = (int)colors.size();
    int num_BConFaceIdx = (int)BConFaceIdx.size();
    int num_BConFaceNum = (int)BConFaceNum.size();
    int numPixels = height * width;
    int numTile = (int)centers.size();
    int total_tile = tile_shape.x * tile_shape.y * tile_shape.z;

    float3 *_vertices, *_centers, *_colors;
    int3 *_faces;
    int2 *_BConFaceNum;
    int *_BConFaceIdx;
    int *_IndexMap, *_visitedTiles, *_valid;
    float *_Ks, *_C2Ws, *_bgdepth;
    float2* _doubleHits;
    float3 *_rays_o, *_rays_d;

    cudaMalloc((void**)&_Ks, sizeof(float)*9*num_camera);
    cudaMemcpy(_Ks, Ks, sizeof(float)*9*num_camera, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&_C2Ws, sizeof(float)*12*num_camera);
    cudaMemcpy(_C2Ws, C2Ws, sizeof(float)*12*num_camera, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&_centers, sizeof(float3)*numTile);
    cudaMemcpy(_centers, centers.data(), sizeof(float3)*numTile, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&_IndexMap, sizeof(int)*total_tile);
    cudaMemcpy(_IndexMap, IndexMap.data(), sizeof(int)*total_tile, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&_vertices, sizeof(float3)*num_vertices);
    cudaMemcpy(_vertices, vertices.data(), sizeof(float3)*num_vertices, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&_faces, sizeof(int3)*num_faces);
    cudaMemcpy(_faces, faces.data(), sizeof(int3)*num_faces, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&_colors, sizeof(float3)*num_colors);
    cudaMemcpy(_colors, colors.data(), sizeof(float3)*num_colors, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&_BConFaceNum, sizeof(int2)*num_BConFaceNum);
    cudaMemcpy(_BConFaceNum, BConFaceNum.data(), sizeof(int2)*num_BConFaceNum, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&_BConFaceIdx, sizeof(int)*num_BConFaceIdx);
    cudaMemcpy(_BConFaceIdx, BConFaceIdx.data(), sizeof(int)*num_BConFaceIdx, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&_visitedTiles, sizeof(int)*max_tracingTile*numPixels);
    cudaMalloc((void**)&_doubleHits, sizeof(float2)*max_tracingTile*numPixels);
    // cudaMalloc((void**)&_firstHitFace, sizeof(int)*max_tracingTile*numPixels);
    cudaMalloc((void**)&_rays_o, sizeof(float3)*numPixels);
    cudaMalloc((void**)&_rays_d, sizeof(float3)*numPixels);
    // cudaMalloc((void**)&_bgcolors, sizeof(float3)*numPixels);
    cudaMalloc((void**)&_bgdepth, sizeof(float)*numPixels);
    cudaMalloc((void**)&_valid, sizeof(int)*numPixels);

    // float3 *bgcolors, *rays_o, *rays_d; 
    float3 *rays_o, *rays_d; 
    float* bgdepth;
    int* valid;
    // cudaHostAlloc( (void**)&bgcolors,sizeof(float3)*numPixels,cudaHostAllocDefault);
    cudaHostAlloc( (void**)&bgdepth,sizeof(float)*numPixels,cudaHostAllocDefault);
    cudaHostAlloc( (void**)&rays_o,sizeof(float3)*numPixels,cudaHostAllocDefault);
    cudaHostAlloc( (void**)&rays_d,sizeof(float3)*numPixels,cudaHostAllocDefault);
    cudaHostAlloc( (void**)&valid,sizeof(int)*numPixels,cudaHostAllocDefault);

    unsigned int n_threads, n_blocks, n_threads2, n_blocks2;
    n_threads = 512;
    n_blocks = min(65535, (numPixels + n_threads - 1) / n_threads);
    n_threads2 = 512;
    n_blocks2 = min(65535, (numPixels*max_tracingTile + n_threads2 - 1) / n_threads2);

    int add_num = numPixels % (n_blocks * n_threads);
    int base_jobs = numPixels / (n_blocks * n_threads);

    int add_num2 = (numPixels*max_tracingTile) % (n_blocks2 * n_threads2);
    int base_jobs2 = (numPixels*max_tracingTile) / (n_blocks2 * n_threads2);

    tqdm bar;
    for (int j=0; j<imgIdxs.size(); j++)
    {
        bar.progress(j, (int)imgIdxs.size());
        int i = imgIdxs[j];
        // cudaMemset(_bgcolors, 0, sizeof(float3)*numPixels);
        cudaMemset(_valid, 0, sizeof(int)*numPixels);

        // get rays
        computing_rays_kernel<<<n_blocks, n_threads>>>(
            height, width, _Ks + i * 9, _C2Ws + i * 12, _rays_o, _rays_d, add_num, base_jobs);

        // get the tileIdx for each ray 
        tracingTile_kernel<<<n_blocks, n_threads>>>(
            height, width, _rays_o, _rays_d, _IndexMap, tile_shape,
            tile_size, max_tracingTile, _visitedTiles, scene_min_corner, add_num, base_jobs);

        ray_doubleHit_inTile_kernel<<<n_blocks2, n_threads2>>>(
            height, width, _rays_o, _rays_d, _vertices, _faces, _centers,
            _BConFaceIdx, _BConFaceNum, tile_size, max_tracingTile, num_block,
            _visitedTiles, _doubleHits, add_num2, base_jobs2);
        
        get_trainData_kernel<<<n_blocks, n_threads>>>(
            trainTileIdx, _rays_o, _rays_d, _vertices, _faces, _visitedTiles,
            _doubleHits, max_tracingTile, _bgdepth, _valid, add_num, base_jobs);

        cudaMemcpy( bgdepth, _bgdepth, sizeof(float)*numPixels, cudaMemcpyDeviceToHost );
        cudaMemcpy( rays_o, _rays_o, sizeof(float3)*numPixels, cudaMemcpyDeviceToHost );
        cudaMemcpy( rays_d, _rays_d, sizeof(float3)*numPixels, cudaMemcpyDeviceToHost );
        cudaMemcpy( valid, _valid, sizeof(int)*numPixels, cudaMemcpyDeviceToHost );


        std::string path = img_path + "/" + std::to_string(i) + ".png";
        cv::Mat gt = cv::imread(path);
        cv::cvtColor(gt, gt, cv::COLOR_BGR2RGB);
        gt.convertTo(gt, CV_32FC3, 1.0f/255.0f);

        path = diffuse_path + "/" + std::to_string(i) + ".png";
        cv::Mat diffuse = cv::imread(path);
        cv::cvtColor(diffuse, diffuse, cv::COLOR_BGR2RGB);
        diffuse.convertTo(diffuse, CV_32FC3, 1.0f/255.0f);

        if (gt.isContinuous() && diffuse.isContinuous())
        {
            float3* gtcolors = (float3*)gt.data;
            float3* diffusecolors = (float3*)diffuse.data;
            for (int b = 0; b<numPixels; b++)
            {
                if (valid[b] == 0) continue;

                data.emplace_back(rays_o[b].x);
                data.emplace_back(rays_o[b].y);
                data.emplace_back(rays_o[b].z);

                data.emplace_back(rays_d[b].x);
                data.emplace_back(rays_d[b].y);
                data.emplace_back(rays_d[b].z);

                data.emplace_back(bgdepth[b]);

                data.emplace_back(gtcolors[b].x);
                data.emplace_back(gtcolors[b].y);
                data.emplace_back(gtcolors[b].z);

                data.emplace_back(diffusecolors[b].x);
                data.emplace_back(diffusecolors[b].y);
                data.emplace_back(diffusecolors[b].z);

            }    
        }else{
            printf("img Not Contiguous!\n");
        }

    
        // cv::Mat img(height, width, CV_32FC3, (float*)bgcolors);
        // img.convertTo(img, CV_8UC3, 255.0f);
        // cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
        // std::string save_path = "bgcolor" + std::to_string(i) + ".png";
        // cv::imwrite(save_path, img);
        if (debug == true)
        {
            std::string save_path = "gtcolor" + std::to_string(i) + ".png";
            float3* gtcolors = (float3*)gt.data;
            for (int b = 0; b<numPixels; b++)
            {
                if (valid[b] == 0) gtcolors[b] *= 0.4;
            }
            gt.convertTo(gt, CV_8UC3, 255.0f);
            cv::cvtColor(gt, gt, cv::COLOR_RGB2BGR);
            cv::imwrite(save_path, gt);

            save_path = "diffuse" + std::to_string(i) + ".png";
            float3* diffusecolors = (float3*)diffuse.data;
            for (int b = 0; b<numPixels; b++)
            {
                if (valid[b] == 0) diffusecolors[b] = make_float3(0,0,0);
            }
            diffuse.convertTo(diffuse, CV_8UC3, 255.0f);
            cv::cvtColor(diffuse, diffuse, cv::COLOR_RGB2BGR);
            cv::imwrite(save_path, diffuse);
        }
    }
    bar.finish();

    cudaFreeHost(bgdepth);
    cudaFreeHost(valid);
    cudaFreeHost(rays_o);
    cudaFreeHost(rays_d);
    cudaFree(_bgdepth);
    cudaFree(_valid);
    cudaFree(_rays_o);
    cudaFree(_rays_d);
    cudaFree(_doubleHits);
    // cudaFree(_firstHitFace);
    cudaFree(_Ks);
    cudaFree(_C2Ws);
    cudaFree(_vertices);
    cudaFree(_faces);
    cudaFree(_colors);
    cudaFree(_centers);
    cudaFree(_BConFaceNum);
    cudaFree(_BConFaceIdx);
    cudaFree(_IndexMap);
    cudaFree(_visitedTiles);

    cudaError_t err = cudaGetLastError();

    if ( err != cudaSuccess )
    {
       printf("CUDA Error: %s\n", cudaGetErrorString(err));       
    }
}

__global__ 
void gen_diffuse_in_tile_kernel(
    const float3* origins,
    const int trainTileIdx,
    const float3* rays_start,
    const float3* rays_dir, 
    const float2* doubleHits, // numPixel x max_tracingTile
    const int* visitedTiles,
    const float3* images,
    const float* depths,
    const float* ms, // N x 12 
    const int max_tracingTile,
    const int num_images,
    const int height, 
    const int width,
    float3* diffuses, // B 
    float* bgdepth,
    int add_num, int base_jobs)
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

    int numPixels = height * width;

    for (int i=0; i<task_num; i++)
    {
        int cur_task_idx = start_task_idx + i;

        const float2* cur_doubleHits = doubleHits + cur_task_idx * max_tracingTile;
        const int* cur_visitedTiles = visitedTiles + cur_task_idx * max_tracingTile;

        float3 rays_o = rays_start[cur_task_idx];
        float3 rays_d = rays_dir[cur_task_idx];

        diffuses[cur_task_idx] = make_float3(0.0f, 0.0f, 0.0f);

        float first_zval = -1;
        for (int j=0; j<max_tracingTile; j++)
        {
            if (cur_doubleHits[j].x != -1 && cur_visitedTiles[j] == trainTileIdx)
            {
                first_zval = cur_doubleHits[j].x;
                break;
            }
        }
        if (first_zval == -1) continue;

        if (bgdepth[cur_task_idx] != -1.0f)
        {
            float3 bgpoint = rays_o + bgdepth[cur_task_idx] * rays_d;
            bool visible = false;
            for (int j=0; j<num_images; j++)
            {
                const float* matrix = ms + j * 12;
                const float* dep = depths + j * numPixels;
                float x = matrix[0] * bgpoint.x + matrix[1] * bgpoint.y + matrix[2] * bgpoint.z + matrix[3];
                float y = matrix[4] * bgpoint.x + matrix[5] * bgpoint.y + matrix[6] * bgpoint.z + matrix[7];
                float z = matrix[8] * bgpoint.x + matrix[9] * bgpoint.y + matrix[10] * bgpoint.z + matrix[11];
                if (z <= 0) continue;

                float px = x / z;
                float py = y / z;
                if (px < 0 || px >= width - 1 || py < 0 || py >= height - 1) continue;

                float d = dep[(int)(py+0.5)*width + (int)(px+0.5)];
                if (abs(z - d) > 0.01f) continue;

                visible = true;
                break;
            }

            if(!visible)
            {
                bgdepth[cur_task_idx] = -1.0f;
            }
        }

        float3 hitpoint = rays_o + first_zval * rays_d;
        float3 color_sum = make_float3(0.0f,0.0f,0.0f);
        // int count = 0;
        float weight = 0.0f;

        for (int j=0; j<num_images; j++)
        {
            float3 proj_origin = origins[j];
            float3 ray_proj = normalize(hitpoint - proj_origin);
            const float* matrix = ms + j * 12;
            const float3* img = images + j * numPixels;
            const float* dep = depths + j * numPixels;
            float x = matrix[0] * hitpoint.x + matrix[1] * hitpoint.y + matrix[2] * hitpoint.z + matrix[3];
            float y = matrix[4] * hitpoint.x + matrix[5] * hitpoint.y + matrix[6] * hitpoint.z + matrix[7];
            float z = matrix[8] * hitpoint.x + matrix[9] * hitpoint.y + matrix[10] * hitpoint.z + matrix[11];
            if (z <= 0) continue;

            float px = x / z;
            float py = y / z;
            if (px < 0 || px >= width - 1 || py < 0 || py >= height - 1) continue;
            
            float d = dep[(int)(py+0.5)*width + (int)(px+0.5)];
            if (abs(z - d) > 0.01f) continue;

            int x0 = (int)px;
            int y0 = (int)py;
            float _x = px - x0;
            float _y = py - y0;
            float w00 = (1.0f-_x) * (1.0f-_y);
            float w11 = _x * _y;
            float w01 = (1.0f-_x) * _y;
            float w10 = _x * (1.0f-_y);
            int idx00 = y0 * width + x0;
            int idx10 = idx00 + 1;
            int idx01 = idx00 + width;
            int idx11 = idx01 + 1;
            float3 c00 = img[idx00];
            float3 c10 = img[idx10];
            float3 c01 = img[idx01];
            float3 c11 = img[idx11];


            float w_color = 1 - dot(normalize(rays_d), ray_proj);
            w_color = w_color * w_color;
            color_sum = color_sum + w_color * (c00 * w00 + c01 * w01 + c10 * w10 + c11 * w11);
            weight += w_color;
            // color_sum = color_sum + (c00 * w00 + c01 * w01 + c10 * w10 + c11 * w11);
            // count += 1;
        }

        if (weight != 0)
        {
            diffuses[cur_task_idx] = color_sum / weight;
        }
    }

}


void get_trainData_v3(
    const int trainTileIdx,
    const std::string img_path,
    const std::string dep_path,
    const std::vector<float3> vertices,
    const std::vector<int3> faces, 
    const std::vector<float3> centers,
    const std::vector<int> IndexMap,
    const std::vector<int> BConFaceIdx,
    const std::vector<int2> BConFaceNum,
    const std::vector<int> imgIdxs,
    const float* Ks, const float* C2Ws,
    const int num_camera,
    const int3 tile_shape,
    const float3 scene_min_corner,
    const float tile_size,
    const int max_tracingTile,
    const int num_block,
    const int height, const int width,
    const int patch_size, 
    const bool debug,
    std::vector<float> &data)
{
    assert(patch_size % 2 == 1);

    int half_patch = patch_size / 2;

    int num_vertices = (int)vertices.size();
    int num_faces = (int)faces.size();
    int num_BConFaceIdx = (int)BConFaceIdx.size();
    int num_BConFaceNum = (int)BConFaceNum.size();
    int numPixels = height * width;
    int numTile = (int)centers.size();
    int total_tile = tile_shape.x * tile_shape.y * tile_shape.z;

    float3 *_vertices, *_centers;
    int3 *_faces;
    int2 *_BConFaceNum;
    int *_BConFaceIdx;
    int *_IndexMap, *_visitedTiles, *_valid;
    float *_Ks, *_C2Ws, *_bgdepth;
    float2* _doubleHits;
    float3 *_rays_o, *_rays_d;

    cudaMalloc((void**)&_Ks, sizeof(float)*9*num_camera);
    cudaMemcpy(_Ks, Ks, sizeof(float)*9*num_camera, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&_C2Ws, sizeof(float)*12*num_camera);
    cudaMemcpy(_C2Ws, C2Ws, sizeof(float)*12*num_camera, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&_centers, sizeof(float3)*numTile);
    cudaMemcpy(_centers, centers.data(), sizeof(float3)*numTile, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&_IndexMap, sizeof(int)*total_tile);
    cudaMemcpy(_IndexMap, IndexMap.data(), sizeof(int)*total_tile, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&_vertices, sizeof(float3)*num_vertices);
    cudaMemcpy(_vertices, vertices.data(), sizeof(float3)*num_vertices, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&_faces, sizeof(int3)*num_faces);
    cudaMemcpy(_faces, faces.data(), sizeof(int3)*num_faces, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&_BConFaceNum, sizeof(int2)*num_BConFaceNum);
    cudaMemcpy(_BConFaceNum, BConFaceNum.data(), sizeof(int2)*num_BConFaceNum, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&_BConFaceIdx, sizeof(int)*num_BConFaceIdx);
    cudaMemcpy(_BConFaceIdx, BConFaceIdx.data(), sizeof(int)*num_BConFaceIdx, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&_visitedTiles, sizeof(int)*max_tracingTile*numPixels);
    cudaMalloc((void**)&_doubleHits, sizeof(float2)*max_tracingTile*numPixels);
    // cudaMalloc((void**)&_firstHitFace, sizeof(int)*max_tracingTile*numPixels);
    cudaMalloc((void**)&_rays_o, sizeof(float3)*numPixels);
    cudaMalloc((void**)&_rays_d, sizeof(float3)*numPixels);
    // cudaMalloc((void**)&_bgcolors, sizeof(float3)*numPixels);
    cudaMalloc((void**)&_bgdepth, sizeof(float)*numPixels);
    cudaMalloc((void**)&_valid, sizeof(int)*numPixels);

    // float3 *bgcolors, *rays_o, *rays_d; 
    float3 *rays_o, *rays_d; 
    float* bgdepth;
    int* valid;
    // cudaHostAlloc( (void**)&bgcolors,sizeof(float3)*numPixels,cudaHostAllocDefault);
    cudaHostAlloc( (void**)&bgdepth,sizeof(float)*numPixels,cudaHostAllocDefault);
    cudaHostAlloc( (void**)&rays_o,sizeof(float3)*numPixels,cudaHostAllocDefault);
    cudaHostAlloc( (void**)&rays_d,sizeof(float3)*numPixels,cudaHostAllocDefault);
    cudaHostAlloc( (void**)&valid,sizeof(int)*numPixels,cudaHostAllocDefault);

    unsigned int n_threads, n_blocks, n_threads2, n_blocks2;
    n_threads = 512;
    n_blocks = min(65535, (numPixels + n_threads - 1) / n_threads);
    n_threads2 = 512;
    n_blocks2 = min(65535, (numPixels*max_tracingTile + n_threads2 - 1) / n_threads2);

    int add_num = numPixels % (n_blocks * n_threads);
    int base_jobs = numPixels / (n_blocks * n_threads);

    int add_num2 = (numPixels*max_tracingTile) % (n_blocks2 * n_threads2);
    int base_jobs2 = (numPixels*max_tracingTile) / (n_blocks2 * n_threads2);
    
    std::vector<float3> images;
    std::vector<float> depths;
    tqdm bar2;
    for (int j=0; j<imgIdxs.size(); j++)
    {
        bar2.progress(j, (int)imgIdxs.size());
        int i = imgIdxs[j];
        std::string path = img_path + "/" + std::to_string(i) + ".png";
        cv::Mat gt = cv::imread(path);
        cv::cvtColor(gt, gt, cv::COLOR_BGR2RGB);
        gt.convertTo(gt, CV_32FC3, 1.0f/255.0f);
        // images.emplace_back(gt);
        float3* gtdata = (float3*)gt.data;
        images.insert(images.end(), gtdata, gtdata+numPixels);

        cnpy::NpyArray arr = cnpy::npy_load(dep_path + "/" + std::to_string(i) + ".npy");
        depths.insert(depths.end(), arr.data<float>(), arr.data<float>()+numPixels);
    }

    bar2.finish();
    printf("Finished loading visable images\n");


    int num_images = (int)imgIdxs.size();
    float *ms, *_ms;
    float3 *origins, *_origins;
    cudaHostAlloc( (void**)&ms,sizeof(float)*num_images*12,cudaHostAllocDefault);
    cudaHostAlloc( (void**)&origins,sizeof(float3)*num_images,cudaHostAllocDefault);
    for (int j=0; j<num_images; j++)
    {
        int i = imgIdxs[j];
        compute_project_matrix(Ks+i*9, C2Ws+i*12, ms+j*12);
        origins[j] = make_float3(C2Ws[i*12+3], C2Ws[i*12+7], C2Ws[i*12+11]);
    }
    cudaMalloc((void**)&_ms, sizeof(float)*num_images*12);
    cudaMemcpy(_ms, ms, sizeof(float)*num_images*12, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&_origins, sizeof(float3)*num_images);
    cudaMemcpy(_origins, origins, sizeof(float3)*num_images, cudaMemcpyHostToDevice);

    printf("Finished computing proj matrix\n");

    float3 *_images;
    float *_depths;
    cudaMalloc((void**)&_images, sizeof(float3)*numPixels*num_images);
    cudaMalloc((void**)&_depths, sizeof(float)*numPixels*num_images);
    cudaMemcpy(_images, images.data(), sizeof(float3)*numPixels*num_images, cudaMemcpyHostToDevice);
    cudaMemcpy(_depths, depths.data(), sizeof(float)*numPixels*num_images, cudaMemcpyHostToDevice);
    float3 *_diffuses, *diffuses;
    cudaHostAlloc( (void**)&diffuses,sizeof(float3)*numPixels, cudaHostAllocDefault);
    cudaMalloc((void**)&_diffuses, sizeof(float3)*numPixels);

    // float *_check_first;
    // float *check_first;
    // cudaHostAlloc( (void**)&check_first,sizeof(float)*numPixels, cudaHostAllocDefault);
    // cudaMalloc((void**)&_check_first, sizeof(float)*numPixels);

    tqdm bar;
    for (int j=0; j<imgIdxs.size(); j++)
    {
        bar.progress(j, (int)imgIdxs.size());
        int i = imgIdxs[j];
        // cudaMemset(_bgcolors, 0, sizeof(float3)*numPixels);
        cudaMemset(_valid, 0, sizeof(int)*numPixels);

        // get rays
        computing_rays_kernel<<<n_blocks, n_threads>>>(
            height, width, _Ks + i * 9, _C2Ws + i * 12, _rays_o, _rays_d, add_num, base_jobs);

        // get the tileIdx for each ray 
        tracingTile_kernel<<<n_blocks, n_threads>>>(
            height, width, _rays_o, _rays_d, _IndexMap, tile_shape,
            tile_size, max_tracingTile, _visitedTiles, scene_min_corner, add_num, base_jobs);

        ray_doubleHit_inTile_kernel<<<n_blocks2, n_threads2>>>(
            height, width, _rays_o, _rays_d, _vertices, _faces, _centers,
            _BConFaceIdx, _BConFaceNum, tile_size, max_tracingTile, num_block,
            _visitedTiles, _doubleHits, add_num2, base_jobs2);
        
        get_trainData_kernel<<<n_blocks, n_threads>>>(
            trainTileIdx, _rays_o, _rays_d, _vertices, _faces, _visitedTiles,
            _doubleHits, max_tracingTile, _bgdepth, _valid, add_num, base_jobs);
        
        gen_diffuse_in_tile_kernel<<<n_blocks, n_threads>>>(
            _origins, trainTileIdx, _rays_o, _rays_d, _doubleHits, _visitedTiles, _images, _depths, _ms, max_tracingTile,
            num_images, height, width, _diffuses, _bgdepth, add_num, base_jobs);

        cudaMemcpy( bgdepth, _bgdepth, sizeof(float)*numPixels, cudaMemcpyDeviceToHost );
        cudaMemcpy( rays_o, _rays_o, sizeof(float3)*numPixels, cudaMemcpyDeviceToHost );
        cudaMemcpy( rays_d, _rays_d, sizeof(float3)*numPixels, cudaMemcpyDeviceToHost );
        cudaMemcpy( valid, _valid, sizeof(int)*numPixels, cudaMemcpyDeviceToHost );
        // numPixels
        cudaMemcpy( diffuses, _diffuses, sizeof(float3)*numPixels, cudaMemcpyDeviceToHost );
        // cudaMemcpy( check_first, _check_first, sizeof(float)*numPixels, cudaMemcpyDeviceToHost );


        // std::string path = img_path + "/" + std::to_string(i) + ".png";
        // cv::Mat gt = cv::imread(path);
        // cv::cvtColor(gt, gt, cv::COLOR_BGR2RGB);
        // gt.convertTo(gt, CV_32FC3, 1.0f/255.0f);
        // cv::Mat gt = images[i];

        // path = diffuse_path + "/" + std::to_string(i) + ".png";
        // cv::Mat diffuse = cv::imread(path);
        // cv::cvtColor(diffuse, diffuse, cv::COLOR_BGR2RGB);
        // diffuse.convertTo(diffuse, CV_32FC3, 1.0f/255.0f);

        float3* gtcolors = images.data() + j * numPixels;
        // float3* diffusecolors = (float3*)diffuse.data;
        for (int y=half_patch; y<height-patch_size; y+=patch_size) // y += patch_size -> y += xxx overlap 
        {
            for (int x=half_patch; x<width-patch_size; x+=patch_size)
            {

                // printf("ppppppppppppppppppppppp\n");
                bool flag = false;
                for (int _y=-half_patch; _y<=half_patch && !flag; _y++)
                {
                    for (int _x=-half_patch; _x<=half_patch && !flag; _x++)
                    {
                        int idx = (y + _y) * width + (x + _x);

                        // printf("x %d y %d x %d y % d height %d width %d idx %d\n",
                        //  x, y, x + _x, y + _y, height, width, idx);

                        if (valid[idx] != 0)
                        {
                            flag = true;
                            break;
                        }
                    }
                }

                // printf("kkkkkkkkkkkkkkkkkkkkkkkkkk\n");

                if (!flag) continue;

                for (int _y=-half_patch; _y<=half_patch; _y++)
                {
                    for (int _x=-half_patch; _x<=half_patch; _x++)
                    {

                        int idx = (y + _y) * width + (x + _x);
                        
                        data.emplace_back(rays_o[idx].x);
                        data.emplace_back(rays_o[idx].y);
                        data.emplace_back(rays_o[idx].z);

                        data.emplace_back(rays_d[idx].x);
                        data.emplace_back(rays_d[idx].y);
                        data.emplace_back(rays_d[idx].z);

                        data.emplace_back(bgdepth[idx]);

                        data.emplace_back(gtcolors[idx].x);
                        data.emplace_back(gtcolors[idx].y);
                        data.emplace_back(gtcolors[idx].z);
                        data.emplace_back(diffuses[idx].x);
                        data.emplace_back(diffuses[idx].y);
                        data.emplace_back(diffuses[idx].z);

                        data.emplace_back((float)valid[idx]);

                    }
                }
            }
        }

        // cv::Mat img(height, width, CV_32FC3, (float*)bgcolors);
        // img.convertTo(img, CV_8UC3, 255.0f);
        // cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
        // std::string save_path = "bgcolor" + std::to_string(i) + ".png";
        // cv::imwrite(save_path, img);
        if (debug == true)
        {
            std::string save_path = "gtcolor" + std::to_string(i) + ".png";
            cv::Mat out_gt = cv::Mat::zeros(height, width, CV_32FC3);
            float3* out_gt_Data = (float3*)out_gt.data;
            for (int b = 0; b<numPixels; b++)
            {
                if (valid[b] == 0) out_gt_Data[b] = gtcolors[b] * 0.4;
                else out_gt_Data[b] = gtcolors[b];
            }
            out_gt.convertTo(out_gt, CV_8UC3, 255.0f);
            cv::cvtColor(out_gt, out_gt, cv::COLOR_RGB2BGR);
            cv::imwrite(save_path, out_gt);

            save_path = "diffuse" + std::to_string(i) + ".png";
            cv::Mat out_diffuse = cv::Mat::zeros(height, width, CV_32FC3);
            float3* out_diffuse_Data = (float3*)out_diffuse.data;
            for (int b = 0; b<numPixels; b++)
            {
                if (valid[b] == 1) out_diffuse_Data[b] = diffuses[b];
            }
            out_diffuse.convertTo(out_diffuse, CV_8UC3, 255.0f);
            cv::cvtColor(out_diffuse, out_diffuse, cv::COLOR_RGB2BGR);
            cv::imwrite(save_path, out_diffuse);

            // save_path = "depth" + std::to_string(i) + ".png";
            // cv::Mat out_depth = cv::Mat::zeros(height, width, CV_32FC1);
            // float* out_depth_Data = (float*)out_depth.data;
            // for (int b = 0; b<numPixels; b++)
            // {
            //     out_depth_Data[b] = check_first[b];
            // }
            // out_depth.convertTo(out_depth, CV_8UC1, 255.0f/100.0f);
            // cv::imwrite(save_path, out_depth);
        }
    }
    bar.finish();
    // cudaFreeHost(check_first);
    // cudaFree(_check_first);
    cudaFreeHost(bgdepth);
    cudaFreeHost(valid);
    cudaFreeHost(rays_o);
    cudaFreeHost(rays_d);
    cudaFreeHost(ms);
    cudaFreeHost(diffuses);
    cudaFreeHost(origins);
    cudaFree(_origins);
    cudaFree(_diffuses);
    cudaFree(_images);
    cudaFree(_depths);
    cudaFree(_ms);
    cudaFree(_bgdepth);
    cudaFree(_valid);
    cudaFree(_rays_o);
    cudaFree(_rays_d);
    cudaFree(_doubleHits);
    cudaFree(_Ks);
    cudaFree(_C2Ws);
    cudaFree(_vertices);
    cudaFree(_faces);
    cudaFree(_centers);
    cudaFree(_BConFaceNum);
    cudaFree(_BConFaceIdx);
    cudaFree(_IndexMap);
    cudaFree(_visitedTiles);

    cudaError_t err = cudaGetLastError();

    if ( err != cudaSuccess )
    {
       printf("CUDA Error: %s\n", cudaGetErrorString(err));       
    }
}


__global__ void render_diffuse_kernel(
    const int trainTileIdx,
    const float3* rays_start, // B x 3
    const float3* rays_dir, // B x 3 
    const int* visitedTiles, // B x Maxtraincing 
    const int* SparseToGroup, 
    const int max_tracingTile,
    const float tile_size,
    const int num_voxel, 
    const float voxel_size,
    const float sample_step, 
    float3* centers, // NT x 3 
    float* voxels, // G x num_voxel x num_voxel x num_voxel x 4
    short* nodes, 
    float3* diffuses,  
    float* depths, 
    int* valid, 
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

    float early_termainate = 0.00001f;

    int voxels_per_tile = num_voxel * num_voxel * num_voxel;

    for (int i=0; i<task_num; i++)
    {
        int cur_task_idx = start_task_idx + i;

        const int* cur_visitedTiles = visitedTiles + cur_task_idx * max_tracingTile;
        float3 rays_o = rays_start[cur_task_idx];
        float3 rays_d = rays_dir[cur_task_idx];

        float3 color_diffuse = make_float3(0,0,0);
        float transparency = 1.0f;

        // bool first_hit = false;
        int firstHit_tileIdx = -1;
        // bool early_break = false;

        for (int j=0; j<max_tracingTile; j++)
        {
            // if (early_break) break;
            if (cur_visitedTiles[j] == -1) break; 
            if (transparency < early_termainate) break;

            int tileIdx = cur_visitedTiles[j];

            int tileIdx_inGroup = SparseToGroup[tileIdx];

            if (tileIdx_inGroup == -1) 
            {
                firstHit_tileIdx = -1;
                break;
            }

            float4* cur_voxels = (float4*)(voxels + tileIdx_inGroup * voxels_per_tile * 4);
            short* cur_nodes = nodes + tileIdx_inGroup * voxels_per_tile;

            float3 tile_center = centers[tileIdx];
            float2 bound = RayAABBIntersection(rays_o, rays_d, tile_center, tile_size/2.0f);
            float3 dilate_corner = tile_center - tile_size / 2.0f - voxel_size;

            float zval = bound.x; 
            while (zval < bound.y)
            {
                float3 pts = rays_o + zval * rays_d;
                int3 pidx = make_int3((pts - dilate_corner) / voxel_size);
                int nidx = pidx.x * num_voxel * num_voxel + pidx.y * num_voxel + pidx.z;
                if (cur_nodes[nidx] != 1)
                {
                    zval += sample_step;
                    continue;
                }
                if (firstHit_tileIdx == -1 && transparency <= 0.5f)
                {
                    // first_hit = true;
                    // if (tileIdx != trainTileIdx)
                    // {
                    //     // early_break = true;
                    //     // break;
                    // }else{
                    depths[cur_task_idx] = zval;
                    firstHit_tileIdx = tileIdx;
                    // }
                }
                float4 rgba = voxels_interpolation(cur_voxels, tile_center, tile_size, num_voxel,
                                                   voxel_size, pts, OVERLAP);
                float sigma = rgba.w; 
                float3 rgb = make_float3(rgba.x, rgba.y, rgba.z); 
                float interval = min(sample_step, bound.y - zval);
                volume_rendering(color_diffuse, transparency, rgb, sigma, interval);
                if (transparency < early_termainate) break; 
                zval += sample_step;
            }
        }
        
        if (firstHit_tileIdx != -1)
        { 
            valid[cur_task_idx] = 1;
        }else{
            valid[cur_task_idx] = 0;
            depths[cur_task_idx] = -1;
        }
        diffuses[cur_task_idx] = color_diffuse;  
    }
}

// For second iteration 
void get_trainData_v4(
    const int trainTileIdx,
    const std::string img_path,
    const std::string diffuse_path, 
    const std::vector<float3> centers,
    const std::vector<int> IndexMap,
    const std::vector<int> imgIdxs,
    const std::vector<int> SparseToGroup,
    const int num_render_tiles, 
    float* voxels, 
    short* nodes, 
    const float* Ks, const float* C2Ws,
    const int num_camera,
    const int3 tile_shape,
    const float3 scene_min_corner,
    const float tile_size,
    const int num_voxel,
    const float voxel_size,
    const float sample_step,
    const int max_tracingTile,
    const int height, const int width,
    const int patch_size, 
    const bool debug,
    std::vector<float> &data)
{
    assert(patch_size % 2 == 1);

    int half_patch = patch_size / 2;

    int numPixels = height * width;
    int numTile = (int)centers.size();
    int total_tile = tile_shape.x * tile_shape.y * tile_shape.z;
    int num_voxels_per_tile = num_voxel * num_voxel * num_voxel;

    float3 *_centers;
    int *_IndexMap, *_visitedTiles, *_valid;
    float *_Ks, *_C2Ws;
    float3 *_rays_o, *_rays_d;
    int *_SparseToGroup;
    float *_voxels, *_depths;
    float3 *_diffuses;
    short* _nodes;
    cudaMalloc((void**)&_voxels, sizeof(float)*4*num_render_tiles*num_voxels_per_tile);
    cudaMemcpy(_voxels, voxels, sizeof(float)*4*num_render_tiles*num_voxels_per_tile, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&_nodes, sizeof(short)*num_render_tiles*num_voxels_per_tile);
    cudaMemcpy(_nodes, nodes, sizeof(short)*num_render_tiles*num_voxels_per_tile, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&_SparseToGroup, sizeof(int)*numTile);
    cudaMemcpy(_SparseToGroup, SparseToGroup.data(), sizeof(int)*numTile, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&_Ks, sizeof(float)*9*num_camera);
    cudaMemcpy(_Ks, Ks, sizeof(float)*9*num_camera, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&_C2Ws, sizeof(float)*12*num_camera);
    cudaMemcpy(_C2Ws, C2Ws, sizeof(float)*12*num_camera, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&_centers, sizeof(float3)*numTile);
    cudaMemcpy(_centers, centers.data(), sizeof(float3)*numTile, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&_IndexMap, sizeof(int)*total_tile);
    cudaMemcpy(_IndexMap, IndexMap.data(), sizeof(int)*total_tile, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&_visitedTiles, sizeof(int)*max_tracingTile*numPixels);
    cudaMalloc((void**)&_diffuses, sizeof(float3)*numPixels);
    cudaMalloc((void**)&_rays_o, sizeof(float3)*numPixels);
    cudaMalloc((void**)&_rays_d, sizeof(float3)*numPixels);
    cudaMalloc((void**)&_valid, sizeof(int)*numPixels);
    cudaMalloc((void**)&_depths, sizeof(float)*numPixels);

    float3 *rays_o, *rays_d; 
    int* valid;
    float3 *diffuses;
    float *depths;
    // cudaHostAlloc( (void**)&bgcolors,sizeof(float3)*numPixels,cudaHostAllocDefault);
    cudaHostAlloc( (void**)&rays_o,sizeof(float3)*numPixels,cudaHostAllocDefault);
    cudaHostAlloc( (void**)&rays_d,sizeof(float3)*numPixels,cudaHostAllocDefault);
    cudaHostAlloc( (void**)&valid,sizeof(int)*numPixels,cudaHostAllocDefault);
    cudaHostAlloc( (void**)&diffuses,sizeof(float3)*numPixels,cudaHostAllocDefault);
    cudaHostAlloc( (void**)&depths,sizeof(float)*numPixels,cudaHostAllocDefault);

    unsigned int n_threads, n_blocks;
    n_threads = 512;
    n_blocks = min(65535, (numPixels + n_threads - 1) / n_threads);
    int add_num = numPixels % (n_blocks * n_threads);
    int base_jobs = numPixels / (n_blocks * n_threads);

    std::vector<float3> images;
    // std::vector<float> mirror_depths;
    // std::vector<float3> diffuse_images;
    tqdm bar2;
    for (int j=0; j<imgIdxs.size(); j++)
    {
        bar2.progress(j, (int)imgIdxs.size());
        int i = imgIdxs[j];
        std::string path = img_path + "/" + std::to_string(i) + ".png";
        cv::Mat gt = cv::imread(path);
        cv::cvtColor(gt, gt, cv::COLOR_BGR2RGB);
        gt.convertTo(gt, CV_32FC3, 1.0f/255.0f);
        // images.emplace_back(gt);
        float3* gtdata = (float3*)gt.data;
        images.insert(images.end(), gtdata, gtdata+numPixels);

        // cnpy::NpyArray arr = cnpy::npy_load(diffuse_path + "/" + std::to_string(i) + ".npy");
        // mirror_depths.insert(mirror_depths.end(), arr.data<float>(), arr.data<float>()+numPixels);

        // path = diffuse_path + "/" + std::to_string(i) + ".png";
        // cv::Mat diffuse = cv::imread(path);
        // cv::cvtColor(diffuse, diffuse, cv::COLOR_BGR2RGB);
        // diffuse.convertTo(diffuse, CV_32FC3, 1.0f/255.0f);
        // float3* diffusedata = (float3*)diffuse.data;
        // diffuse_images.insert(diffuse_images.end(), diffusedata, diffusedata+numPixels);

    }

    bar2.finish();
    printf("Finished loading visable images and mirror depths\n");

    tqdm bar;
    for (int j=0; j<imgIdxs.size(); j++)
    {
        bar.progress(j, (int)imgIdxs.size());
        int i = imgIdxs[j];
        // cudaMemset(_bgcolors, 0, sizeof(float3)*numPixels);
        cudaMemset(_valid, 0, sizeof(int)*numPixels);

        // get rays
        computing_rays_kernel<<<n_blocks, n_threads>>>(
            height, width, _Ks + i * 9, _C2Ws + i * 12, _rays_o, _rays_d, add_num, base_jobs);

        // get the tileIdx for each ray 
        tracingTile_kernel<<<n_blocks, n_threads>>>(
            height, width, _rays_o, _rays_d, _IndexMap, tile_shape,
            tile_size, max_tracingTile, _visitedTiles, scene_min_corner, add_num, base_jobs);

        render_diffuse_kernel<<<n_blocks, n_threads>>>(
            trainTileIdx, _rays_o, _rays_d, _visitedTiles, _SparseToGroup,
            max_tracingTile, tile_size, num_voxel, voxel_size, 
            sample_step, _centers, _voxels, _nodes, _diffuses, 
            _depths, _valid, base_jobs, add_num);
        
        cudaMemcpy( rays_o, _rays_o, sizeof(float3)*numPixels, cudaMemcpyDeviceToHost );
        cudaMemcpy( rays_d, _rays_d, sizeof(float3)*numPixels, cudaMemcpyDeviceToHost );
        cudaMemcpy( valid, _valid, sizeof(int)*numPixels, cudaMemcpyDeviceToHost );
        cudaMemcpy( depths, _depths, sizeof(float)*numPixels, cudaMemcpyDeviceToHost );
        cudaMemcpy( diffuses, _diffuses, sizeof(float3)*numPixels, cudaMemcpyDeviceToHost );


        float3* gtcolors = images.data() + j * numPixels;
        // float* mdepth = mirror_depths.data() + j * numPixels;
        for (int y=half_patch; y<height-patch_size; y+=half_patch)
        {
            for (int x=half_patch; x<width-patch_size; x+=half_patch)
            {

                bool flag = false;
                for (int _y=-half_patch; _y<=half_patch && !flag; _y++)
                {
                    for (int _x=-half_patch; _x<=half_patch && !flag; _x++)
                    {
                        int idx = (y + _y) * width + (x + _x);
                        if (valid[idx] != 0)
                        {
                            flag = true;
                            break;
                        }
                    }
                }

                // printf("kkkkkkkkkkkkkkkkkkkkkkkkkk\n");

                if (!flag) continue;

                for (int _y=-half_patch; _y<=half_patch; _y++)
                {
                    for (int _x=-half_patch; _x<=half_patch; _x++)
                    {

                        int idx = (y + _y) * width + (x + _x);
                        
                        data.emplace_back(rays_o[idx].x);
                        data.emplace_back(rays_o[idx].y);
                        data.emplace_back(rays_o[idx].z);

                        data.emplace_back(rays_d[idx].x);
                        data.emplace_back(rays_d[idx].y);
                        data.emplace_back(rays_d[idx].z);

                        data.emplace_back(depths[idx]);

                        data.emplace_back(gtcolors[idx].x);
                        data.emplace_back(gtcolors[idx].y);
                        data.emplace_back(gtcolors[idx].z);
                        data.emplace_back(diffuses[idx].x);
                        data.emplace_back(diffuses[idx].y);
                        data.emplace_back(diffuses[idx].z);

                        data.emplace_back((float)valid[idx]);
                        // data.emplace_back(mdepth[idx]);

                    }
                }
            }
        }

        // cv::Mat img(height, width, CV_32FC3, (float*)bgcolors);
        // img.convertTo(img, CV_8UC3, 255.0f);
        // cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
        // std::string save_path = "bgcolor" + std::to_string(i) + ".png";
        // cv::imwrite(save_path, img);
        // if ( trainTileIdx != 215) continue;
        if (debug == true)
        {
            std::string save_path = "gtcolor" + std::to_string(i) + ".png";
            cv::Mat out_gt = cv::Mat::zeros(height, width, CV_32FC3);
            float3* out_gt_Data = (float3*)out_gt.data;
            for (int b = 0; b<numPixels; b++)
            {
                if (valid[b] == 0) out_gt_Data[b] = gtcolors[b] * 0.4;
                else out_gt_Data[b] = gtcolors[b];
            }
            out_gt.convertTo(out_gt, CV_8UC3, 255.0f);
            cv::cvtColor(out_gt, out_gt, cv::COLOR_RGB2BGR);
            cv::imwrite(save_path, out_gt);

            save_path = "diffuse" + std::to_string(i) + ".png";
            cv::Mat out_diffuse = cv::Mat::zeros(height, width, CV_32FC3);
            float3* out_diffuse_Data = (float3*)out_diffuse.data;
            for (int b = 0; b<numPixels; b++)
            {
                if (valid[b] == 1) out_diffuse_Data[b] = diffuses[b];
            }
            out_diffuse.convertTo(out_diffuse, CV_8UC3, 255.0f);
            cv::cvtColor(out_diffuse, out_diffuse, cv::COLOR_RGB2BGR);
            cv::imwrite(save_path, out_diffuse);
            // float3* diffusecolors = (float3*)diffuse.data;
            // for (int b = 0; b<numPixels; b++)
            // {
            //     if (valid[b] == 0) diffusecolors[b] = make_float3(0,0,0);
            // }
            // diffuse.convertTo(diffuse, CV_8UC3, 255.0f);
            // cv::cvtColor(diffuse, diffuse, cv::COLOR_RGB2BGR);
            // cv::imwrite(save_path, diffuse);
        }
    }
    bar.finish();

    cudaFreeHost(valid);
    cudaFreeHost(rays_o);
    cudaFreeHost(rays_d);
    cudaFreeHost(diffuses);
    cudaFreeHost(depths);
    cudaFree(_valid);
    cudaFree(_rays_o);
    cudaFree(_rays_d);
    cudaFree(_diffuses);
    cudaFree(_SparseToGroup);
    cudaFree(_depths);
    // cudaFree(_firstHitFace);
    cudaFree(_Ks);
    cudaFree(_C2Ws);
    cudaFree(_centers);
    cudaFree(_IndexMap);
    cudaFree(_visitedTiles);

    cudaError_t err = cudaGetLastError();

    if ( err != cudaSuccess )
    {
       printf("CUDA Error: %s\n", cudaGetErrorString(err));       
    }
}

void get_VisImg(
    const std::vector<float3> vertices,
    const std::vector<int3> faces, 
    const std::vector<float3> centers,
    const std::vector<int> IndexMap,
    const std::vector<int> BConFaceIdx,
    const std::vector<int2> BConFaceNum,
    const float* Ks, const float* C2Ws,
    const int num_camera,
    const int3 tile_shape,
    const float3 scene_min_corner,
    const float tile_size,
    const int max_tracingTile,
    const int num_block,
    const int height, const int width,
    std::vector<int> &VisImg) // NumTile x NumCamera
{
    int num_vertices = (int)vertices.size();
    int num_faces = (int)faces.size();
    int num_BConFaceIdx = (int)BConFaceIdx.size();
    int num_BConFaceNum = (int)BConFaceNum.size();
    int numPixels = height * width;
    int numTile = (int)centers.size();
    int total_tile = tile_shape.x * tile_shape.y * tile_shape.z;

    float3 *_vertices, *_centers;
    int3 *_faces;
    int2 *_BConFaceNum;
    int *_BConFaceIdx;
    int *_IndexMap, *_visitedTiles, *_firstHitFace;
    float *_Ks, *_C2Ws, *_firstHits;
    float3 *_rays_o, *_rays_d;
    int *_VisImg;

    cudaMalloc((void**)&_VisImg, sizeof(int)*numTile*num_camera);
    cudaMemset(_VisImg, 0, sizeof(int)*numTile*num_camera);

    cudaMalloc((void**)&_Ks, sizeof(float)*9*num_camera);
    cudaMemcpy(_Ks, Ks, sizeof(float)*9*num_camera, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&_C2Ws, sizeof(float)*12*num_camera);
    cudaMemcpy(_C2Ws, C2Ws, sizeof(float)*12*num_camera, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&_centers, sizeof(float3)*numTile);
    cudaMemcpy(_centers, centers.data(), sizeof(float3)*numTile, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&_IndexMap, sizeof(int)*total_tile);
    cudaMemcpy(_IndexMap, IndexMap.data(), sizeof(int)*total_tile, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&_vertices, sizeof(float3)*num_vertices);
    cudaMemcpy(_vertices, vertices.data(), sizeof(float3)*num_vertices, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&_faces, sizeof(int3)*num_faces);
    cudaMemcpy(_faces, faces.data(), sizeof(int3)*num_faces, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&_BConFaceNum, sizeof(int2)*num_BConFaceNum);
    cudaMemcpy(_BConFaceNum, BConFaceNum.data(), sizeof(int2)*num_BConFaceNum, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&_BConFaceIdx, sizeof(int)*num_BConFaceIdx);
    cudaMemcpy(_BConFaceIdx, BConFaceIdx.data(), sizeof(int)*num_BConFaceIdx, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&_visitedTiles, sizeof(int)*max_tracingTile*numPixels);
    cudaMalloc((void**)&_firstHits, sizeof(float)*max_tracingTile*numPixels);
    cudaMalloc((void**)&_firstHitFace, sizeof(int)*max_tracingTile*numPixels);
    cudaMalloc((void**)&_rays_o, sizeof(float3)*numPixels);
    cudaMalloc((void**)&_rays_d, sizeof(float3)*numPixels);

    unsigned int n_threads, n_blocks, n_threads2, n_blocks2;
    n_threads = 512;
    n_blocks = min(65535, (numPixels + n_threads - 1) / n_threads);
    n_threads2 = 512;
    n_blocks2 = min(65535, (numPixels*max_tracingTile + n_threads2 - 1) / n_threads2);

    int add_num = numPixels % (n_blocks * n_threads);
    int base_jobs = numPixels / (n_blocks * n_threads);

    int add_num2 = (numPixels*max_tracingTile) % (n_blocks2 * n_threads2);
    int base_jobs2 = (numPixels*max_tracingTile) / (n_blocks2 * n_threads2);

    tqdm bar;

    for (int i=0; i<num_camera; i++)
    {
        bar.progress(i, num_camera);
        // printf("process %d/%d\n", i, num_camera);
        cudaMemset(_firstHits, 0, sizeof(float)*max_tracingTile*numPixels);

        // get rays
        computing_rays_kernel<<<n_blocks, n_threads>>>(
            height, width, _Ks + i * 9, _C2Ws + i * 12, _rays_o, _rays_d, add_num, base_jobs);
        // get the tileIdx for each ray 
        tracingTile_kernel<<<n_blocks, n_threads>>>(
            height, width, _rays_o, _rays_d, _IndexMap, tile_shape,
            tile_size, max_tracingTile, _visitedTiles, scene_min_corner, add_num, base_jobs);
        ray_firstHit_inTile_kernel<<<n_blocks2, n_threads2>>>(
            height, width, _rays_o, _rays_d, _vertices, _faces, _centers,
            _BConFaceIdx, _BConFaceNum, tile_size, max_tracingTile, num_block, 
            _visitedTiles, _firstHits, _firstHitFace, add_num2, base_jobs2);
        
        get_VisImg_kernel<<<n_blocks, n_threads>>>(
            i, num_camera, _visitedTiles, _firstHits, max_tracingTile,
            _VisImg, add_num, base_jobs);
    }
    bar.finish();

    cudaMemcpy( VisImg.data(), _VisImg, sizeof(int)*numTile*num_camera, cudaMemcpyDeviceToHost );


    cudaFree(_firstHits);
    cudaFree(_VisImg);
    cudaFree(_firstHitFace);
    cudaFree(_Ks);
    cudaFree(_C2Ws);
    cudaFree(_vertices);
    cudaFree(_faces);
    cudaFree(_centers);
    cudaFree(_BConFaceNum);
    cudaFree(_BConFaceIdx);
    cudaFree(_IndexMap);
    cudaFree(_visitedTiles);

    cudaError_t err = cudaGetLastError();

    if ( err != cudaSuccess )
    {
       printf("CUDA Error: %s\n", cudaGetErrorString(err));       
    }
}


__global__
void render_TileScene_kernel(
    const float4* voxels, 
    const float3* centers,
    const float3 scene_min_corner,
    const int3 tile_shape,
    const float tile_size,
    const int* IndexMap,
    const int num_voxel,
    const float voxel_size,
    const float* K,
    const float* C2W,
    const int height, 
    const int width,
    const int max_tracingTile,
    const int Nsamples,
    int* visitedTiles,
    float2* bounds,
    float3* frame,
    int add_num, int base_jobs)
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
    int voxels_per_tile = num_voxel * num_voxel * num_voxel;
    float3 zeros = make_float3(0.0f, 0.0f, 0.0f);
    float3 scene_size = make_float3(tile_shape) * tile_size;
    for (int i=0; i<task_num; i++)
    {
        int cur_task_idx = start_task_idx + i;

        int* cur_visitedTiles = visitedTiles + cur_task_idx * max_tracingTile;
        float2* cur_bounds = bounds + cur_task_idx * max_tracingTile;

        for (int j=0; j<max_tracingTile; j++)
        {
            cur_visitedTiles[j] = -1;
        }


        int _y = cur_task_idx / width;
        int _x = cur_task_idx % width;

        float3 ray_o, ray_d;
        get_rays(_x, _y, K, C2W, ray_o, ray_d);

        float3 ray_o_local = ray_o - scene_min_corner;

        if (ray_o_local.x < 0 || ray_o_local.x >= scene_size.x ||
            ray_o_local.y < 0 || ray_o_local.y >= scene_size.y ||
            ray_o_local.z < 0 || ray_o_local.z >= scene_size.z )
        {
            float2 bound = RayAABBIntersection(ray_o, ray_d, 
                                scene_min_corner + scene_size / 2.0f, scene_size / 2.0f);
            ray_o_local = ray_o_local + bound.x * ray_d;
        }

        ray_o_local = clamp(ray_o_local, zeros, scene_size - 0.000001f);
        voxel_traversal(&ray_o_local.x, &ray_d.x, IndexMap, cur_visitedTiles, max_tracingTile, 
                        tile_shape.x, tile_shape.y, tile_shape.z, tile_size);
    
        
        // float length = 0.0f;
        // int num_tracing_tile = -1;
        // for (int j=0; j<max_tracingTile; j++)
        // {
        //     if (cur_visitedTiles[j] == -1)
        //     {
        //         num_tracing_tile = j;
        //         break;
        //     }

        //     int tileIdx = cur_visitedTiles[j];
        //     float3 tile_center = centers[tileIdx];

        //     // float actual_size = half_size - voxel_size / 2.0f;
        //     cur_bounds[j] = RayAABBIntersection(ray_o, ray_d, tile_center, half_size);

        //     length += (cur_bounds[j].y - cur_bounds[j].x);
        // }

        // if (length == 0) continue;

        float3 color = make_float3(0,0,0); 
        float transparency = 1.0f; 
        float weights = 0.0f;
        for (int j=0; j<max_tracingTile; j++)
        {
            if (cur_visitedTiles[j] == -1) break; 

            int tileIdx = cur_visitedTiles[j];
            float3 tile_center = centers[tileIdx];
            cur_bounds[j] = RayAABBIntersection(ray_o, ray_d, tile_center, half_size);

            const float4* cur_voxels = voxels + tileIdx * voxels_per_tile;

            float near = cur_bounds[j].x;
            float far = cur_bounds[j].y;

            near = near + 0.00001f;
            far = far - 0.00001f;

            if (near > far) continue;

            // int sample_num = (int)(far - near) / (voxel_size * 0.5);
            // if (sample_num < 1) sample_num = 1;
            int sample_num = (int)(Nsamples * (far - near) / tile_size);

            // if (cur_task_idx == 58984)
            // {
            //     printf("sample_num: %d\n", sample_num);
            // }
            sample_num = sample_num >= 2? sample_num:2;

            float interval = (far - near) / (sample_num-1); 
            float zval = near;
            for (int k=0; k<sample_num-1; k++)
            {
                float4 rgba = voxels_interpolation(cur_voxels, tile_center, tile_size, num_voxel,
                                                    voxel_size, ray_o + zval * ray_d, OVERLAP);
                
                // if (cur_task_idx == 58984)
                // {
                //     float3 tile_corner = tile_center - tile_size / 2.0f;
                //     float3 fidx = ((ray_o + zval * ray_d -tile_corner) + voxel_size / 2.0f) / voxel_size;
                //     int3 idx = make_int3(fidx+0.5f);
                //     printf("fidx: %f %f %f\nidx %d %d %d  zval: %f far: %f\ncolor: %f %f %f %f\n\n",
                //     fidx.x, fidx.y, fidx.z,
                //     idx.x, idx.y, idx.z, zval, far, rgba.x, rgba.y, rgba.z, rgba.w);
                // }

                float3 rgb = make_float3(rgba.x, rgba.y, rgba.z);

                rgb = 1.0f / (expf(-1.0f * rgb)+1.0f);
                // rgb = clamp(rgb, 0,1);

                float sigma = rgba.w;
                float alpha = 1.0f - expf(-1.0f * sigma * interval);

                // float alpha = rgba.w;
                float weight = alpha * transparency;
                color = color + weight * rgb;
                weights += weight;

                // if (cur_task_idx == 172676 && sigma != 0)
                // {
                //     float3 R = 1.0f / (expf(inverse_rgb) + 1.0f);
                //     float3 tile_corner = tile_center - tile_size / 2.0f;
                //     float3 fidx = ((ray_o + zval * ray_d -tile_corner) + voxel_size / 2.0f) / voxel_size;
                //     printf("k %d tileIdx %d near: %f far: %f fidx: %f %f %f\nrgb: %f %f %f %f %f %f\ninverse: %f %f %f\nexpf:%f %f %f\ncolor: %f %f %f transparency: %f weights: %f alpha: %f interval: %f sigma: %f\n\n", 
                //     k, tileIdx, near, far, fidx.x, fidx.y, fidx.z, rgb.x, rgb.y, rgb.z, rgba.x, rgba.y, rgba.z,
                //     inverse_rgb.x, inverse_rgb.y, inverse_rgb.z,
                //     R.x, R.y,R.z,
                //     color.x, color.y, color.z, transparency, weights, alpha, interval, sigma);
                // }

                transparency = transparency * (1-alpha);

                zval += interval;

                if (transparency <= 0.0000001) break;

            }
            if (transparency <= 0.0000001) break;
        }

        // if (cur_task_idx == 58984)
        // {
        //     printf("color: %f %f %f\n", color.x, color.y, color.z);
        // }

        // if (weights != 0) color = color / weights;
        frame[cur_task_idx] = color;

    }
}


void render_TileScene(
    const std::vector<float4> voxels,
    const std::vector<float3> centers,
    const float3 scene_min_corner,
    const int3 tile_shape,
    const float tile_size,
    const int max_tracingTile,
    const std::vector<int> IndexMap,
    const int num_voxel,
    const float voxel_size,
    const float* K,
    const float* C2W,
    const int height, 
    const int width,
    const int Nsamples,
    std::vector<float3> &frame)
{

    int numPixels = height * width;
    int numTile = (int)centers.size();
    int total_voxels = (int)voxels.size();
    int num_IndexMap = (int)IndexMap.size();
    assert(total_voxels == numTile * num_voxel * num_voxel * num_voxel);

    float4 *_voxels;
    float3 *_centers, *_frame;
    int *_IndexMap, *_visitedTiles;
    float *_K, *_C2W;
    float2 *_bounds;


    cudaMalloc((void**)&_K, sizeof(float)*9);
    cudaMemcpy(_K, K, sizeof(float)*9, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&_C2W, sizeof(float)*12);
    cudaMemcpy(_C2W, C2W, sizeof(float)*12, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&_voxels, sizeof(float4)*total_voxels);
    cudaMemcpy(_voxels, voxels.data(), sizeof(float4)*total_voxels, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&_centers, sizeof(float3)*numTile);
    cudaMemcpy(_centers, centers.data(), sizeof(float3)*numTile, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&_IndexMap, sizeof(int)*num_IndexMap);
    cudaMemcpy(_IndexMap, IndexMap.data(), sizeof(int)*num_IndexMap, cudaMemcpyHostToDevice);


    cudaMalloc((void**)&_frame, sizeof(float3)*numPixels);
    cudaMemset(_frame, 0, sizeof(float3)*numPixels);


    cudaMalloc((void**)&_visitedTiles, sizeof(int)*numPixels*max_tracingTile);
    cudaMalloc((void**)&_bounds, sizeof(float2)*numPixels*max_tracingTile);


    unsigned int n_threads, n_blocks;
    n_threads = 512;
    n_blocks = min(65535, (numPixels + n_threads - 1) / n_threads);

    int add_num = numPixels % (n_blocks * n_threads);
    int base_jobs = numPixels / (n_blocks * n_threads);

    render_TileScene_kernel<<<n_blocks, n_threads>>>(
        _voxels, _centers, scene_min_corner, tile_shape, tile_size,
        _IndexMap, num_voxel, voxel_size, _K, _C2W, height, width, max_tracingTile,
        Nsamples, _visitedTiles, _bounds, _frame, add_num, base_jobs);
        
    cudaMemcpy( frame.data(), _frame, sizeof(float3)*numPixels, cudaMemcpyDeviceToHost );


    cudaFree(_frame);
    cudaFree(_visitedTiles);
    cudaFree(_bounds);
    cudaFree(_K);
    cudaFree(_C2W);
    cudaFree(_voxels);
    cudaFree(_centers);
    cudaFree(_IndexMap);

    cudaError_t err = cudaGetLastError();

    if ( err != cudaSuccess )
    {
       printf("CUDA Error: %s\n", cudaGetErrorString(err));       
    }
}


__global__ void gen_voxels_kernel(
    const float3* vertices,
    const int3* faces,
    const float3* colors,
    const float3* centers,
    const float3* sizes,
    const int* ConFaceIdx,
    const int* ConFaceStart,
    const int* ConFaceNum,
    const float voxel_size,
    const int num_voxel,
    float4* voxels,
    float* min_dis,
    int add_num, int base_jobs)
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

    int total_voxels = num_voxel * num_voxel * num_voxel;
    float3 voxel_dim = make_float3(voxel_size,voxel_size,voxel_size);

    for (int i=0; i<task_num; i++)
    {
        int cur_task_idx = start_task_idx + i;
        float3 tile_center = centers[cur_task_idx];
        float3 half_size = sizes[cur_task_idx] / 2.0f;
        float3 min_corner = tile_center - half_size;
        float4* cur_voxels = voxels + cur_task_idx * total_voxels;
        float* cur_min_dis = min_dis + cur_task_idx * total_voxels;

        for (int j=0; j<total_voxels; j++)
        {
            cur_min_dis[j] = INF;
        }
        int start_face = ConFaceStart[cur_task_idx];
        int face_num = ConFaceNum[cur_task_idx];
        for (int j=start_face; j<start_face + face_num; j++)
        {
            int3 vidx = faces[ConFaceIdx[j]];
            // printf("%d %d %d %d\n", cur_task_idx, vidx.x, vidx.y, vidx.z);
            float3 A = vertices[vidx.x];
            float3 B = vertices[vidx.y];
            float3 C = vertices[vidx.z];

            float3 min_c = fminf(fminf(A,B),C) - min_corner;
            float3 max_c = fmaxf(fmaxf(A,B),C) - min_corner;

            int3 min_idx = make_int3(min_c / voxel_size);
            int3 max_idx = make_int3(max_c / voxel_size);

            min_idx = clamp(min_idx, 0, num_voxel-1);
            max_idx = clamp(max_idx, 0, num_voxel-1);

            for (int x=min_idx.x; x<=max_idx.x; x++)
            {
                for (int y=min_idx.y; y<=max_idx.y; y++)
                {
                    for (int z=min_idx.z; z<=max_idx.z; z++)
                    {
                        int voxel_idx = x * num_voxel * num_voxel + y * num_voxel + z;
                        float3 pos = make_float3((float)x,(float)y,(float)z);

                        float3 voxel_center = min_corner + pos * voxel_size + voxel_size / 2.0f;
                        float dis_A = norm(A - voxel_center);
                        float dis_B = norm(B - voxel_center);
                        float dis_C = norm(C - voxel_center);

                        float dis = dis_A; int idx = vidx.x;
                        if (dis_B < dis) 
                        {
                            dis = dis_B;
                            idx = vidx.y;
                        }
                        if (dis_C < dis)
                        {
                            dis = dis_C;
                            idx = vidx.z;
                        }
                        if (dis < cur_min_dis[voxel_idx])
                        {
                            cur_min_dis[voxel_idx] = dis;
                            cur_voxels[voxel_idx].x = colors[idx].x;
                            cur_voxels[voxel_idx].y = colors[idx].y;
                            cur_voxels[voxel_idx].z = colors[idx].z;
                        }
                        if (cur_voxels[voxel_idx].w == 1) continue;
                        if (AABB_triangle_intersection(voxel_center, voxel_dim, A, B, C))
                        {
                            cur_voxels[voxel_idx].w = 1.0f;
                        }
                    }
                }
            }
        }
    }
}


void gen_voxels(
    const std::vector<float3> vertices,
    const std::vector<int3> faces, 
    const std::vector<float3> colors,
    const std::vector<float3> centers,
    const std::vector<float3> sizes,
    const std::vector<int> ConFaceIdx,
    const std::vector<int> ConFaceNum,
    const float voxel_size,
    const int num_voxel,
    std::vector<float4> &voxels)
{
    int num_vertices = (int)vertices.size();
    int num_faces = (int)faces.size();
    int num_colors = (int)colors.size();
    int num_centers = (int)centers.size();
    int num_sizes = (int)sizes.size();
    int num_ConFaceIdx = (int)ConFaceIdx.size();
    int num_ConFaceNum = (int)ConFaceNum.size();

    std::vector<int> ConFaceStart;
    int count = 0;
    for (int i=0; i<ConFaceNum.size(); i++)
    {
        ConFaceStart.emplace_back(count);
        count += ConFaceNum[i];
    }

    float3 *_vertices, *_colors, *_centers, *_sizes;
    int3 *_faces;
    int *_ConFaceIdx, *_ConFaceNum, *_ConFaceStart;
    float4 *_voxels;

    cudaMalloc((void**)&_vertices, sizeof(float3)*num_vertices);
    cudaMemcpy(_vertices, vertices.data(), sizeof(float3)*num_vertices, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&_faces, sizeof(int3)*num_faces);
    cudaMemcpy(_faces, faces.data(), sizeof(int3)*num_faces, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&_centers, sizeof(float3)*num_centers);
    cudaMemcpy(_centers, centers.data(), sizeof(float3)*num_centers, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&_sizes, sizeof(float3)*num_sizes);
    cudaMemcpy(_sizes, sizes.data(), sizeof(float3)*num_sizes, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&_colors, sizeof(float3)*num_colors);
    cudaMemcpy(_colors, colors.data(), sizeof(float3)*num_colors, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&_ConFaceStart, sizeof(int)*num_centers);
    cudaMemcpy(_ConFaceStart, ConFaceStart.data(), sizeof(int)*num_centers, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&_ConFaceNum, sizeof(int)*num_centers);
    cudaMemcpy(_ConFaceNum, ConFaceNum.data(), sizeof(int)*num_centers, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&_ConFaceIdx, sizeof(int)*num_ConFaceIdx);
    cudaMemcpy(_ConFaceIdx, ConFaceIdx.data(), sizeof(int)*num_ConFaceIdx, cudaMemcpyHostToDevice);

    int total_voxels = num_centers * num_voxel * num_voxel * num_voxel;
    cudaMalloc((void**)&_voxels, sizeof(float4)*total_voxels);
    cudaMemset(_voxels, 0, sizeof(float4)*total_voxels);
    
    float *_min_dis;
    cudaMalloc((void**)&_min_dis, sizeof(float)*total_voxels);


    unsigned int n_threads, n_blocks;
    n_threads = 256;
    n_blocks = min(65535, (num_centers + n_threads - 1) / n_threads);

    int add_num = num_centers % (n_blocks * n_threads);
    int base_jobs = num_centers / (n_blocks * n_threads);

    gen_voxels_kernel<<<n_blocks, n_threads>>>(
        _vertices, _faces, _colors, _centers, _sizes, _ConFaceIdx, _ConFaceStart, _ConFaceNum, 
        voxel_size, num_voxel, _voxels,
        _min_dis, add_num, base_jobs);

    cudaMemcpy(voxels.data(), _voxels, sizeof(float4)*total_voxels, cudaMemcpyDeviceToHost);

    cudaFree(_min_dis);
    cudaFree(_voxels);
    cudaFree(_vertices);
    cudaFree(_faces);
    cudaFree(_centers);
    cudaFree(_sizes);
    cudaFree(_colors);
    cudaFree(_ConFaceStart);
    cudaFree(_ConFaceNum);
    cudaFree(_ConFaceIdx);

    cudaError_t err = cudaGetLastError();

    if ( err != cudaSuccess )
    {
       printf("CUDA Error: %s\n", cudaGetErrorString(err));       
    }
}

