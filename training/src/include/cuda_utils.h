#ifndef _CUDA_UTILS_H
#define _CUDA_UTILS_H

#include <cuda.h>
#include <cuda_runtime.h>
#include "cutil_math.h"

#define INF 100000000
#define THRESH 0.00001


inline __device__
void volume_rendering(float3 &color, float &transparency, float3 rgb, float sigma, float step)
{
    float alpha = 1.0f - expf(-1.0f * sigma * step);
    float weight = alpha * transparency;
    color = color + weight * rgb;
    transparency = transparency * (1.0f - alpha);
}


inline __device__
void volume_rendering(float3 &color_diffuse, float3 &color_specular, 
                    float &transparency, float3 diffuse, float3 specular, float sigma, float step)
{
    float alpha = 1.0f - expf(-1.0f * sigma * step);
    float weight = alpha * transparency;
    color_diffuse = color_diffuse + weight * diffuse;
    color_specular = color_specular + weight * specular;
    transparency = transparency * (1.0f - alpha);
}





inline __device__ __host__ void compute_project_matrix(const float* K, const float* C2W, float* m)
{
    float fx = K[0], fy = K[4], cx = K[2], cy = K[5];
    float t1 = -1.0f * (C2W[0] * C2W[3] + C2W[4] * C2W[7] + C2W[8] * C2W[11]);
    float t2 = -1.0f * (C2W[1] * C2W[3] + C2W[5] * C2W[7] + C2W[9] * C2W[11]);
    float t3 = -1.0f * (C2W[2] * C2W[3] + C2W[6] * C2W[7] + C2W[10] * C2W[11]);

    // std::cout << t1 << " " << t2 << " " << t3 << std::endl;

    m[0] = fx * C2W[0] + cx * C2W[2];
    m[1] = fx * C2W[4] + cx * C2W[6];
    m[2] = fx * C2W[8] + cx * C2W[10];
    m[3] = fx * t1 + cx * t3;
    m[4] = fy * C2W[1] + cy * C2W[2];
    m[5] = fy * C2W[5] + cy * C2W[6];
    m[6] = fy * C2W[9] + cy * C2W[10];
    m[7] = fy * t2 + cy * t3;
    m[8] = C2W[2];
    m[9] = C2W[6];
    m[10] = C2W[10];
    m[11] = t3;
}


inline __device__ __host__ void inverse_z_sample_bound(
    float* zvals,
    float near, float far,
    int Nsamples)
{
    float inv_near = 1.0f / near;
    float inv_far = 1.0f / far;
    float inv_bound = inv_far - inv_near;

    float step = 1.0f / (Nsamples-1);
    for (int i=0; i<Nsamples; i++)
    {
        zvals[i] = 1.0f / (step * i * inv_bound + inv_near);
    }
}

inline __device__ __host__ void uniform_sample_bound(
    float* zvals,
    float near, float far,
    int Nsamples)
{
    float interval = (far - near) / (Nsamples - 1);
    for (int i=0; i<Nsamples; i++)
    {
        zvals[i] = near + i * interval;
    }
}

inline __device__ __host__ void uniform_sample_bound_v2(
    float* zvals,
    float near, float far,
    int Nsamples)
{
    float interval = (far - near) / Nsamples;
    for (int i=0; i<Nsamples; i++)
    {
        zvals[i] = near + i * interval;
    }
}

inline __device__ __host__ void uniform_sample_bound_v3(
    float* zvals,
    float* dists,
    float near, float far,
    int Nsamples)
{
    float interval = (far - near) / Nsamples;
    for (int i=0; i<Nsamples; i++)
    {
        zvals[i] = near + i * interval;
        dists[i] = interval;
    }
}


inline __device__ __host__ float3 barycentric_weight(
    const float3 A, 
    const float3 B, 
    const float3 C,
    const float3 P)
{
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
    return make_float3(u,v,w);
}

inline __device__ int get_index(int Mx, int My, int Mz, int* voxel_index)
{
    return voxel_index[0] * My * Mz + voxel_index[1] * Mz + voxel_index[2]; 
}


inline __device__ void get_rays(int _x, int _y, const float* K, const float* C2W,
                                float3 &rays_o, float3 &rays_d)
{
    float x = (1.0f * _x - K[2]) / K[0];
    float y = (1.0f * _y - K[5]) / K[4];
    
    rays_d.x = C2W[0] * x + C2W[1] * y + C2W[2];
    rays_d.y = C2W[4] * x + C2W[5] * y + C2W[6];
    rays_d.z = C2W[8] * x + C2W[9] * y + C2W[10];
    rays_o.x = C2W[3];
    rays_o.y = C2W[7];
    rays_o.z = C2W[11];
}


inline __device__ void voxel_traversal(
    const float* ray_o, const float* ray_d, const int* IndexMap, int* visited, 
    const int K, int Mx, int My, int Mz, const float voxel_size)
{

    int current_voxel[3];
    for (int i=0; i<3; i++)
    {
        current_voxel[i] = (int)floor(ray_o[i] / voxel_size);
    }

    int step[3];
    for (int i=0; i<3; i++)
    {
        if (ray_d[i] >= 0) step[i] = 1;
        else step[i] = -1;
    }

    float next_boundary[3];
    for (int i=0; i<3; i++)
    {
        next_boundary[i] = (current_voxel[i] + step[i]) * voxel_size; 
        if (step[i] < 0) next_boundary[i] += voxel_size;
    }

    float tMax[3];
    for (int i=0; i<3; i++)
    {
        tMax[i] = ray_d[i]!=0? ((next_boundary[i] - ray_o[i]) / ray_d[i]) : INF;
    }

    float tDelta[3];
    for (int i=0; i<3; i++)
    {
        tDelta[i] = ray_d[i]!=0? (voxel_size / ray_d[i] * step[i]) : INF; 
    }
    
    int idx = 0;
    int index;

    index = IndexMap[get_index(Mx, My, Mz, current_voxel)];
    if (index != -1)
    {
        visited[idx] = index;
        idx += 1;
    }

    while (idx < K)
    {
        if (tMax[0] < tMax[1]){
            if (tMax[0] < tMax[2])
            {
                current_voxel[0] += step[0];
                tMax[0] += tDelta[0];
            }else{
                current_voxel[2] += step[2];
                tMax[2] += tDelta[2];
            }
        }else{
            if (tMax[1] < tMax[2])
            {
                current_voxel[1] += step[1];
                tMax[1] += tDelta[1];
            }else{
                current_voxel[2] += step[2];
                tMax[2] += tDelta[2];
            }
        }
        if (current_voxel[0] < 0 || current_voxel[1] < 0 || current_voxel[2] < 0 || 
            current_voxel[0] >= Mx || current_voxel[1] >= My || current_voxel[2] >= Mz)
        {
            break;
        }
        index = IndexMap[get_index(Mx, My, Mz, current_voxel)];
        if (index != -1)
        {
            visited[idx] = index;
            idx += 1;
        }
    }
}


inline __device__ void voxel_traversal_sparse(
    const float* ray_o, const float* ray_d, const bool* flag, int* visited,
    const int K, int Mx, int My, int Mz, const float voxel_size)
{

    int current_voxel[3];
    for (int i=0; i<3; i++)
    {
        current_voxel[i] = (int)floor(ray_o[i] / voxel_size);
    }

    int step[3];
    for (int i=0; i<3; i++)
    {
        if (ray_d[i] >= 0) step[i] = 1;
        else step[i] = -1;
    }

    float next_boundary[3];
    for (int i=0; i<3; i++)
    {
        next_boundary[i] = (current_voxel[i] + step[i]) * voxel_size; 
        if (step[i] < 0) next_boundary[i] += voxel_size;
    }

    float tMax[3];
    for (int i=0; i<3; i++)
    {
        tMax[i] = ray_d[i]!=0? ((next_boundary[i] - ray_o[i]) / ray_d[i]) : INF;
    }

    float tDelta[3];
    for (int i=0; i<3; i++)
    {
        tDelta[i] = ray_d[i]!=0? (voxel_size / ray_d[i] * step[i]) : INF; 
    }
    
    int idx = 0;
    int index;

    index = get_index(Mx, My, Mz, current_voxel);
    if (flag[index] == true)
    {
        visited[idx] = index;
        idx += 1;
    }

    while (idx < K)
    {
        if (tMax[0] < tMax[1]){
            if (tMax[0] < tMax[2])
            {
                current_voxel[0] += step[0];
                tMax[0] += tDelta[0];
            }else{
                current_voxel[2] += step[2];
                tMax[2] += tDelta[2];
            }
        }else{
            if (tMax[1] < tMax[2])
            {
                current_voxel[1] += step[1];
                tMax[1] += tDelta[1];
            }else{
                current_voxel[2] += step[2];
                tMax[2] += tDelta[2];
            }
        }
        if (current_voxel[0] < 0 || current_voxel[1] < 0 || current_voxel[2] < 0 || 
            current_voxel[0] >= Mx || current_voxel[1] >= My || current_voxel[2] >= Mz)
        {
            break;
        }

        index = get_index(Mx, My, Mz, current_voxel);
        if (flag[index] == true)
        {
            visited[idx] = index;
            idx += 1;
        }
    }
}


inline __device__ void voxel_traversal_dense(
    const float* ray_o, const float* ray_d, int* visited, 
    const int K, int Mx, int My, int Mz, const float voxel_size)
{

    int current_voxel[3];
    for (int i=0; i<3; i++)
    {
        current_voxel[i] = (int)floor(ray_o[i] / voxel_size);
    }

    int step[3];
    for (int i=0; i<3; i++)
    {
        if (ray_d[i] >= 0) step[i] = 1;
        else step[i] = -1;
    }

    float next_boundary[3];
    for (int i=0; i<3; i++)
    {
        next_boundary[i] = (current_voxel[i] + step[i]) * voxel_size; 
        if (step[i] < 0) next_boundary[i] += voxel_size;
    }

    float tMax[3];
    for (int i=0; i<3; i++)
    {
        tMax[i] = ray_d[i]!=0? ((next_boundary[i] - ray_o[i]) / ray_d[i]) : INF;
    }

    float tDelta[3];
    for (int i=0; i<3; i++)
    {
        tDelta[i] = ray_d[i]!=0? (voxel_size / ray_d[i] * step[i]) : INF; 
    }
    
    int idx = 0;
    int index;

    index = get_index(Mx, My, Mz, current_voxel);
    visited[idx] = index;
    idx += 1;

    while (idx < K)
    {
        if (tMax[0] < tMax[1]){
            if (tMax[0] < tMax[2])
            {
                current_voxel[0] += step[0];
                tMax[0] += tDelta[0];
            }else{
                current_voxel[2] += step[2];
                tMax[2] += tDelta[2];
            }
        }else{
            if (tMax[1] < tMax[2])
            {
                current_voxel[1] += step[1];
                tMax[1] += tDelta[1];
            }else{
                current_voxel[2] += step[2];
                tMax[2] += tDelta[2];
            }
        }
        if (current_voxel[0] < 0 || current_voxel[1] < 0 || current_voxel[2] < 0 || 
            current_voxel[0] >= Mx || current_voxel[1] >= My || current_voxel[2] >= Mz)
        {
            break;
        }
        index = get_index(Mx, My, Mz, current_voxel);
        visited[idx] = index;
        idx += 1;
    }
}



inline __device__ __host__ float RayTriangleIntersection(
    float3 ori,  float3 dir,
    float3 v0, float3 v1, float3 v2,
    float blur)
{
    float3 v0v1 = v1 - v0;
    float3 v0v2 = v2 - v0;
    float3 v0O = ori - v0;

    float3 dir_crs_v0v2 = cross(dir, v0v2);

    float det = dot(v0v1, dir_crs_v0v2);

    if (det < 0.00000001f) return -INF;

    det = 1.0f / det;

    float u = dot(v0O, dir_crs_v0v2) * det;

    if ((u < 0.0f - blur) || (u > 1.0f + blur))
        return -INF;
  
    float3 v0O_crs_v0v1 = cross(v0O, v0v1);

    float v = dot(dir, v0O_crs_v0v1) * det;
    if ((v < 0.0f - blur) || (v > 1.0f + blur))
        return -INF;
      
    if (((u + v) < 0.0f - blur) || ((u + v) > 1.0f + blur))
        return -INF;
  
    float t = dot(v0v2, v0O_crs_v0v1) * det;
    return t;
}

inline __host__ __device__ bool AABB_triangle_intersection(
    float3 center, float3 size, 
    float3 A, float3 B, float3 C)
{
    // 把AABB移动到世界坐标原点，同时移动三角形
    A = A - center;
    B = B - center;
    C = C - center;

    float3 half_size = size / 2.0f;

    // 三角形的最大corner小于AABB的最小corner 或者 三角形的最小corner大于AABB的最大corner 
    float3 max_tri_corner = fmaxf(fmaxf(A,B),C);
    if (max_tri_corner.x <= (-1 * half_size.x)) return false;
    if (max_tri_corner.y <= (-1 * half_size.y)) return false;
    if (max_tri_corner.z <= (-1 * half_size.z)) return false;


    float3 min_tri_corner = fminf(fminf(A,B),C);
    if (min_tri_corner.x >= half_size.x) return false;
    if (min_tri_corner.y >= half_size.y) return false;
    if (min_tri_corner.z >= half_size.z) return false;

    // 计算三角形三边向量
    float3 AB = B - A;
    float3 BC = C - B;
    float3 CA = A - C;
    float3 side[3] = {AB, BC, CA};
    // AABB的normal
    float3 eye[3] = {make_float3(1,0,0), make_float3(0,1,0), make_float3(0,0,1)};

    
    // 计算 9 个 axis 
    for (int i=0; i<3; i++)
    {
        for (int j=0; j<3; j++)
        {
            float3 axis = cross(eye[i], side[j]);
            // 把三角形三个点投影到轴上
            float3 p = make_float3(dot(A,axis),dot(B,axis),dot(C,axis));

            // AABB 已经被移动到原点
            float r = dot(size, fabs(axis));
            float max_p = fmax(fmax(p.x, p.y), p.z);
            float min_p = fmin(fmin(p.x, p.y), p.z);
            if (max(-1.0f * max_p, min_p) > r) return false;
        }
    }

    for (int i=0; i<3; i++)
    {
        float3 axis = eye[i];
        // 把三角形三个点投影到轴上
        float3 p = make_float3(dot(A,axis),dot(B,axis),dot(C,axis));

        // AABB 已经被移动到原点
        float r = dot(size, fabs(axis));
        float max_p = fmax(fmax(p.x, p.y), p.z);
        float min_p = fmin(fmin(p.x, p.y), p.z);
        if (max(-1.0f * max_p, min_p) > r) return false;
    }

    float3 axis = cross(AB, BC);
    // 把三角形三个点投影到轴上
    float3 p = make_float3(dot(A,axis),dot(B,axis),dot(C,axis));
    // AABB 已经被移动到原点
    float r = dot(size, fabs(axis));
    float max_p = fmax(fmax(p.x, p.y), p.z);
    float min_p = fmin(fmin(p.x, p.y), p.z);
    if (max(-1.0f * max_p, min_p) > r) return false;

    return true;
}


inline __device__ float2 RayAABBIntersection(
  const float3 ori,
  const float3 dir,
  const float3 center,
  float half_voxel) {

  float f_low = 0;
  float f_high = 100000.;
  float f_dim_low, f_dim_high, temp, inv_ray_dir, start, aabb;

  for (int d = 0; d < 3; ++d) {  
    switch (d) {
      case 0:
        inv_ray_dir = safe_divide(1.0f, dir.x); start = ori.x; aabb = center.x; break;
      case 1:
        inv_ray_dir = safe_divide(1.0f, dir.y); start = ori.y; aabb = center.y; break;
      case 2:
        inv_ray_dir = safe_divide(1.0f, dir.z); start = ori.z; aabb = center.z; break;
    }
  
    f_dim_low  = (aabb - half_voxel - start) * inv_ray_dir;
    f_dim_high = (aabb + half_voxel - start) * inv_ray_dir;
  
    // Make sure low is less than high
    if (f_dim_high < f_dim_low) {
      temp = f_dim_low;
      f_dim_low = f_dim_high;
      f_dim_high = temp;
    }

    // If this dimension's high is less than the low we got then we definitely missed.
    if (f_dim_high < f_low) {
      return make_float2(-1.0f, -1.0f);
    }
  
    // Likewise if the low is less than the high.
    if (f_dim_low > f_high) {
      return make_float2(-1.0f, -1.0f);
    }
      
    // Add the clip from this dimension to the previous results 
    f_low = (f_dim_low > f_low) ? f_dim_low : f_low;
    f_high = (f_dim_high < f_high) ? f_dim_high : f_high;
    
    if (f_low > f_high) {
      return make_float2(-1.0f, -1.0f);
    }
  }
  return make_float2(f_low, f_high);
}

inline __device__ float2 RayAABBIntersection(
  const float3 ori,
  const float3 dir,
  const float3 center,
  float3 half_size) {

  float f_low = 0;
  float f_high = 100000.;
  float f_dim_low, f_dim_high, temp, inv_ray_dir, start, aabb, half_voxel;

  for (int d = 0; d < 3; ++d) {  
    switch (d) {
      case 0:
        inv_ray_dir = safe_divide(1.0f, dir.x); start = ori.x; aabb = center.x; half_voxel = half_size.x; break;
      case 1:
        inv_ray_dir = safe_divide(1.0f, dir.y); start = ori.y; aabb = center.y; half_voxel = half_size.y; break;
      case 2:
        inv_ray_dir = safe_divide(1.0f, dir.z); start = ori.z; aabb = center.z; half_voxel = half_size.z; break;
    }
  
    f_dim_low  = (aabb - half_voxel - start) * inv_ray_dir;
    f_dim_high = (aabb + half_voxel - start) * inv_ray_dir;
  
    // Make sure low is less than high
    if (f_dim_high < f_dim_low) {
      temp = f_dim_low;
      f_dim_low = f_dim_high;
      f_dim_high = temp;
    }

    // If this dimension's high is less than the low we got then we definitely missed.
    if (f_dim_high < f_low) {
      return make_float2(-1.0f, -1.0f);
    }
  
    // Likewise if the low is less than the high.
    if (f_dim_low > f_high) {
      return make_float2(-1.0f, -1.0f);
    }
      
    // Add the clip from this dimension to the previous results 
    f_low = (f_dim_low > f_low) ? f_dim_low : f_low;
    f_high = (f_dim_high < f_high) ? f_dim_high : f_high;
    
    if (f_low > f_high) {
      return make_float2(-1.0f, -1.0f);
    }
  }
  return make_float2(f_low, f_high);
}


#endif 