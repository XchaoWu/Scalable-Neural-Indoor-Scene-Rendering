#ifndef _INTERPOLATION_H
#define _INTERPOLATION_H
#include <cuda.h>
#include <cuda_runtime.h>
#include "cutil_math.h"

enum INTER_TYPE {NEAREST, BILINEAR, TRILINEAR, OVERLAP, ALPHA};

/*
Interpolation Functions 
*/

template<class T>
inline __device__ __host__ T interpolate1D(T v1, T v2, float x)
{
    return v1 * (1.0f-x) +  v2 * x; 
}


template<class T>
inline __device__ __host__ T interpolate2D(
    T v1, T v2, T v3, T v4, float x, float y)
{
    T s = interpolate1D(v1, v2, x);
    T t = interpolate1D(v3, v4, x);
    return interpolate1D(s,t,y);
}

template<class T>
inline __device__ __host__ T interpolate3D(
    T v1, T v2, T v3, T v4, 
    T v5, T v6, T v7, T v8,
    float x, float y, float z)
{
    T s = interpolate2D(v1,v2,v3,v4, x, y);
    T t = interpolate2D(v5,v6,v7,v8, x, y);
    return interpolate1D(s, t, z);
}


inline __device__ __host__ 
float4 trilinear_base(
    const float4* voxels,
    const int Mx, const int My, const int Mz,
    const float3 point)
{
    int MYZ = My * Mz;

    int x0 = int(point.x);
    int y0 = int(point.y);
    int z0 = int(point.z);
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    int z1 = z0 + 1;

    float x = point.x - x0;
    float y = point.y - y0;
    float z = point.z - z0;

    int idx_x0y0z0 = x0 * MYZ + y0 * Mz + z0;
    int idx_x1y0z0 = x1 * MYZ + y0 * Mz + z0;
    int idx_x0y1z0 = x0 * MYZ + y1 * Mz + z0;
    int idx_x1y1z0 = x1 * MYZ + y1 * Mz + z0;
    int idx_x0y0z1 = x0 * MYZ + y0 * Mz + z1;
    int idx_x1y0z1 = x1 * MYZ + y0 * Mz + z1;
    int idx_x0y1z1 = x0 * MYZ + y1 * Mz + z1;
    int idx_x1y1z1 = x1 * MYZ + y1 * Mz + z1;

    if (x < 0) x = 0;

    if (y < 0) y = 0;

    if (z < 0) z = 0;
 
    if (x1 >= Mx)
    {
        if (y1 >= My)
        {
            if (z1 >= Mz)
            {
                return voxels[idx_x0y0z0];
            }else{
                return interpolate1D(voxels[idx_x0y0z0],voxels[idx_x0y0z1],z);
            }
        }else{
            if (z1 >= Mz)
            {
                return interpolate1D(voxels[idx_x0y0z0],voxels[idx_x0y1z0],y);
            }else{
                return interpolate2D(voxels[idx_x0y0z0], voxels[idx_x0y1z0],
                                     voxels[idx_x0y0z1], voxels[idx_x0y1z1],y,z);
            }
        }
    }else{
        if (y1 >= My)
        {
            if (z1 >= Mz)
            {
                return interpolate1D(voxels[idx_x0y0z0],voxels[idx_x1y0z0],x);
            }else{
                return interpolate2D(voxels[idx_x0y0z0], voxels[idx_x1y0z0],
                                     voxels[idx_x0y0z1], voxels[idx_x1y0z1],x,z);
            }
        }else{
            if (z1 >= Mz)
            {
                return interpolate2D(voxels[idx_x0y0z0], voxels[idx_x1y0z0],
                                     voxels[idx_x0y1z0], voxels[idx_x1y1z0],x,y); 
            }else{
                return interpolate3D(voxels[idx_x0y0z0], voxels[idx_x1y0z0], voxels[idx_x0y1z0], voxels[idx_x1y1z0],
                                     voxels[idx_x0y0z1], voxels[idx_x1y0z1], voxels[idx_x0y1z1], voxels[idx_x1y1z1],
                                     x,y,z);
            }
        }
    }
}



inline __device__ __host__
float4 overlap_interpolation(
    const float4* voxels,
    const float3 tile_center, 
    const float tile_size,
    const int num_voxel,
    const float voxel_size,
    const float3 p)
{
    float3 tile_corner = tile_center - tile_size / 2.0f;
    // overlap 1 voxel
    float3 fidx = (p - tile_corner) / voxel_size + 0.5f;

    return trilinear_base(voxels, num_voxel, num_voxel, num_voxel, fidx);
}


inline __device__ __host__
float4 nearest_interpolation(
    const float4* voxels,
    const float3 tile_center, 
    const float tile_size,
    const int num_voxel,
    const float voxel_size,
    const float3 p)
{
    float3 tile_corner = tile_center - tile_size / 2.0f;
    float3 fidx = ((p -tile_corner) + voxel_size / 2.0f) / voxel_size;
    int3 idx = make_int3(fidx+0.5f);
    int voxelIdx = idx.x * num_voxel * num_voxel +  idx.y * num_voxel + idx.z;
    return voxels[voxelIdx];
}


inline __device__ __host__
float4 trilinear_interpolation(
    const float4* voxels,
    const float3 tile_center, 
    const float tile_size,
    const int num_voxel,
    const float voxel_size,
    const float3 p)
{
    float3 tile_corner = tile_center - tile_size / 2.0f;
    float3 fidx = ((p - tile_corner) - voxel_size / 2.0f) / voxel_size;
    return trilinear_base(voxels, num_voxel, num_voxel, num_voxel, fidx);
}

/*
插值3D点P 
*/
inline __device__ __host__ 
float4 voxels_interpolation(
    const float4* voxels,
    const float3 tile_center, 
    const float tile_size,
    const int num_voxel,
    const float voxel_size,
    const float3 p,
    INTER_TYPE mode)
{
    if (mode == OVERLAP){
        return overlap_interpolation(voxels, tile_center, tile_size, num_voxel, voxel_size, p);
    }else if (mode == NEAREST){
        return nearest_interpolation(voxels, tile_center, tile_size, num_voxel, voxel_size, p);
    }else if (mode == TRILINEAR){
        return trilinear_interpolation(voxels, tile_center, tile_size, num_voxel, voxel_size, p);
    }
    return make_float4(0,0,0,0);
}

#endif 