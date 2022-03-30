
#ifndef HALF_OPE_H
#define HALF_OPE_H

#include "cuda_fp16.h"

// --------------- half 

inline __device__ __half operator+(__half a, __half b)
{
    return __hadd(a, b);
}

inline __device__ __half operator+(__half a, float b)
{
    return __hadd(a, __float2half(b));
}

inline __device__ void operator+=(__half &a, __half s)
{
    a = __hadd(a, s);
}
inline __device__ __half operator-(__half a, __half b)
{
    return __hsub(a, b);
}
inline __device__ __half operator-(__half a, float b)
{
    return __hsub(a, __float2half(b));
}

inline __device__ __half operator-(float b, __half a)
{
    return __hsub(__float2half(b), a);
}
inline __device__ void operator-=(__half &a, __half s)
{
    a = __hsub(a, s);
}
inline __device__ __half operator*(__half a, __half b)
{
    return __hmul(a, b);
}
inline __device__ __half operator*(__half a, float b)
{
    return __hmul(a, __float2half(b));
}
inline __device__ __half operator*(float b, __half a)
{
    return __hmul(a, __float2half(b));
}
inline __device__ void operator*=(__half &a, __half s)
{
    a = __hmul(a, s);
}

inline __device__ __half operator/(__half a, __half b)
{
    return __hdiv(a, b);
}
inline __device__ __half operator/(__half a, float b)
{
    return __hdiv(a, __float2half(b));
}
inline __device__ void operator/=(__half &a, __half s)
{
    a = __hdiv(a, s);
}

struct half3
{
    __half x,y,z;
};

inline __device__ half3 make_half3(__half x, __half y, __half z)
{
    half3 h;
    h.x = x;
    h.y = y;
    h.z = z; 
    return h;
}

inline __device__ half3 make_half3(float x, float y, float z)
{
    half3 h;
    h.x = __float2half(x);
    h.y = __float2half(y);
    h.z = __float2half(z); 
    return h;
}

inline __device__ half3 operator+(half3 a, half3 b)
{
    return make_half3(a.x+b.x, a.y+b.y, a.z+b.z);
}
inline __device__ void operator+=(half3& a, half3 b)
{
    a.x = a.x + b.x;
    a.y = a.y + b.y;
    a.z = a.z + b.z;
}
inline __device__ half3 operator-(half3 a, half3 b)
{
    return make_half3(a.x-b.x, a.y-b.y, a.z-b.z);
}

inline __device__ void operator-=(half3& a, half3 b)
{
    a.x = a.x - b.x;
    a.y = a.y - b.y;
    a.z = a.z - b.z;
}


inline __device__ half3 operator*(half3 a, half3 b)
{
    return make_half3(a.x*b.x, a.y*b.y, a.z*b.z);
}
inline __device__ half3 operator*(half3 a, float b)
{
    return make_half3(a.x*b, a.y*b, a.z*b);
}
inline __device__ half3 operator*(half a, half3 b)
{
    return make_half3(a*b.x, a*b.y, a*b.z);
}

inline __device__ void operator*=(half3& a, half3 b)
{
    a.x = a.x * b.x;
    a.y = a.y * b.y;
    a.z = a.z * b.z;
}
inline __device__ void operator*=(float b, half3& a)
{
    a.x = a.x * b;
    a.y = a.y * b;
    a.z = a.z * b;
}
inline __device__ half3 operator/(half3 a, half3 b)
{
    return make_half3(a.x/b.x, a.y/b.y, a.z/b.z);
}
inline __device__ void operator/=(half3& a, half3 b)
{
    a.x = a.x / b.x;
    a.y = a.y / b.y;
    a.z = a.z / b.z;
}


struct half4
{
    __half x,y,z,w;
 
};


inline __device__ half4 make_half4(__half x, __half y, __half z, __half w)
{
    half4 h;
    h.x = x;
    h.y = y;
    h.z = z; 
    h.w = w;
    return h;
}

inline __device__ half4 operator+(half4 a, half4 b)
{
    return make_half4(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w);
}
inline __device__ void operator+=(half4& a, half4 b)
{
    a.x = a.x + b.x;
    a.y = a.y + b.y;
    a.z = a.z + b.z;
    a.w = a.w + b.w;
}
inline __device__ half4 operator-(half4 a, half4 b)
{
    return make_half4(a.x-b.x, a.y-b.y, a.z-b.z, a.w-b.w);
}
inline __device__ void operator-=(half4& a, half4 b)
{
    a.x = a.x - b.x;
    a.y = a.y - b.y;
    a.z = a.z - b.z;
    a.w = a.w - b.w;
}


inline __device__ half4 operator*(half4 a, half4 b)
{
    return make_half4(a.x*b.x, a.y*b.y, a.z*b.z, a.w*b.w);
}
inline __device__ half4 operator*(half4 a, float b)
{
    return make_half4(a.x*b, a.y*b, a.z*b, a.w*b);
}

inline __device__ void operator*=(half4& a, half4 b)
{
    a.x = a.x * b.x;
    a.y = a.y * b.y;
    a.z = a.z * b.z;
    a.w = a.w * b.w;
}
inline __device__ half4 operator/(half4 a, half4 b)
{
    return make_half4(a.x/b.x, a.y/b.y, a.z/b.z, a.w/b.w);
}
inline __device__ void operator/=(half4& a, half4 b)
{
    a.x = a.x / b.x;
    a.y = a.y / b.y;
    a.z = a.z / b.z;
    a.w = a.w / b.w;
}

inline __device__ half4 half_interpolate1D(half4 v1, half4 v2, float x)
{
    return v1 * (1.0f-x) +  v2 * x; 
}


inline __device__ half4 half_interpolate2D(
    half4 v1, half4 v2, half4 v3, half4 v4, float x, float y)
{
    half4 s = half_interpolate1D(v1, v2, x);
    half4 t = half_interpolate1D(v3, v4, x);
    return half_interpolate1D(s,t,y);
}

inline __device__ half4 half_interpolate3D(
    half4 v1, half4 v2, half4 v3, half4 v4, 
    half4 v5, half4 v6, half4 v7, half4 v8,
    float x, float y, float z)
{
    half4 s = half_interpolate2D(v1,v2,v3,v4, x, y);
    half4 t = half_interpolate2D(v5,v6,v7,v8, x, y);
    return half_interpolate1D(s, t, z);
}

inline __device__ half4 trilinear_base(
    const half4* voxels,
    const short* IndexMap,
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

    short idx_x0y0z0 = IndexMap[x0 * MYZ + y0 * Mz + z0];
    short idx_x1y0z0 = IndexMap[x1 * MYZ + y0 * Mz + z0];
    short idx_x0y1z0 = IndexMap[x0 * MYZ + y1 * Mz + z0];
    short idx_x1y1z0 = IndexMap[x1 * MYZ + y1 * Mz + z0];
    short idx_x0y0z1 = IndexMap[x0 * MYZ + y0 * Mz + z1];
    short idx_x1y0z1 = IndexMap[x1 * MYZ + y0 * Mz + z1];
    short idx_x0y1z1 = IndexMap[x0 * MYZ + y1 * Mz + z1];
    short idx_x1y1z1 = IndexMap[x1 * MYZ + y1 * Mz + z1];

    if (x < 0) x = 0;

    if (y < 0) y = 0;

    if (z < 0) z = 0;


    return half_interpolate3D(voxels[idx_x0y0z0], voxels[idx_x1y0z0], voxels[idx_x0y1z0], voxels[idx_x1y1z0],
                              voxels[idx_x0y0z1], voxels[idx_x1y0z1], voxels[idx_x0y1z1], voxels[idx_x1y1z1],
                              x,y,z);
}


#endif 