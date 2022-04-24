#ifndef _CUDA_MLP_H
#define _CUDA_MLP_H

#include <cuda.h> 
#include <cuda_runtime.h>
#include "cutil_math.h"

#define C0 0.28209479177387814
#define C1 0.4886025119029199
#define C20 1.0925484305920792
#define C21 -1.0925484305920792
#define C22 0.31539156525252005
#define C23 -1.0925484305920792
#define C24 0.5462742152960396
#define C30 -0.5900435899266435
#define C31 2.890611442640554
#define C32 -0.4570457994644658
#define C33 0.3731763325901154
#define C34 -0.4570457994644658
#define C35 1.445305721320277
#define C36 -0.5900435899266435
#define C40 2.5033429417967046
#define C41 -1.7701307697799304
#define C42 0.9461746957575601
#define C43 -0.6690465435572892
#define C44 0.10578554691520431
#define C45 -0.6690465435572892
#define C46 0.47308734787878004
#define C47 -1.7701307697799304
#define C48 0.6258357354491761

enum ACTIVATION {SIGMOID, RELU, TANH, LEAKYRELU};

inline __device__ __host__
float3 inference_SH_nodiffuse(const int deg, const float3 rays_d, float3* coeffi)
{
    // rays_d 需要 normalize 
    assert(deg <= 4 and deg >= 0);
    float3 result = make_float3(0.0f, 0.0f, 0.0f);
    if (deg > 0)
    {
        float x = rays_d.x, y = rays_d.y, z = rays_d.z;
        result = result - C1 * y * coeffi[0] + C1 * z * coeffi[1] - C1 * x * coeffi[2];
        if (deg > 1)
        {
            float xx = x*x, yy = y*y, zz = z*z;
            float xy = x*y, yz = y*z, xz = x*z;
            result = result + 
                     C20 * xy * coeffi[3] + 
                     C21 * yz * coeffi[4] + 
                     C22 * (2 * zz - xx - yy) * coeffi[5] + 
                     C23 * xz * coeffi[6] + 
                     C24 * (xx - yy) * coeffi[7];
            if (deg > 2)
            {
                result = result +
                        C30 * y * (3 * xx - yy) * coeffi[8] +
                        C31 * xy * z * coeffi[9] +
                        C32 * y * (4 * zz - xx - yy)* coeffi[10] +
                        C33 * z * (2 * zz - 3 * xx - 3 * yy) * coeffi[11] +
                        C34 * x * (4 * zz - xx - yy) * coeffi[12] +
                        C35 * z * (xx - yy) * coeffi[13] +
                        C36 * x * (xx - 3 * yy) * coeffi[14];
                
                if (deg > 3)
                {
                    result = result + C40 * xy * (xx - yy) * coeffi[15] +
                            C41 * yz * (3 * xx - yy) * coeffi[16] +
                            C42 * xy * (7 * zz - 1) * coeffi[17] +
                            C43 * yz * (7 * zz - 3) * coeffi[18] +
                            C44 * (zz * (35 * zz - 30) + 3) * coeffi[19] +
                            C45 * xz * (7 * zz - 3) * coeffi[20] +
                            C46 * (xx - yy) * (7 * zz - 1) * coeffi[21] +
                            C47 * xz * (xx - 3 * yy) * coeffi[22] +
                            C48 * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * coeffi[23];
                }
            }
                      
        }
    }
    return result;
}


inline __device__ __host__
float3 inference_SH(const int deg, const float3 rays_d, float3* coeffi)
{
    // rays_d 需要 normalize 
    assert(deg <= 4 and deg >= 0);

    float3 result = C0 * coeffi[0];
    if (deg > 0)
    {
        float x = rays_d.x, y = rays_d.y, z = rays_d.z;
        result = result - C1 * y * coeffi[1] + C1 * z * coeffi[2] - C1 * x * coeffi[3];
        if (deg > 1)
        {
            float xx = x*x, yy = y*y, zz = z*z;
            float xy = x*y, yz = y*z, xz = x*z;
            result = result + 
                     C20 * xy * coeffi[4] + 
                     C21 * yz * coeffi[5] + 
                     C22 * (2 * zz - xx - yy) * coeffi[6] + 
                     C23 * xz * coeffi[7] + 
                     C24 * (xx - yy) * coeffi[8];
            if (deg > 2)
            {
                result = result +
                        C30 * y * (3 * xx - yy) * coeffi[9] +
                        C31 * xy * z * coeffi[10] +
                        C32 * y * (4 * zz - xx - yy)* coeffi[11] +
                        C33 * z * (2 * zz - 3 * xx - 3 * yy) * coeffi[12] +
                        C34 * x * (4 * zz - xx - yy) * coeffi[13] +
                        C35 * z * (xx - yy) * coeffi[14] +
                        C36 * x * (xx - 3 * yy) * coeffi[15];
                
                if (deg > 3)
                {
                    result = result + C40 * xy * (xx - yy) * coeffi[16] +
                            C41 * yz * (3 * xx - yy) * coeffi[17] +
                            C42 * xy * (7 * zz - 1) * coeffi[18] +
                            C43 * yz * (7 * zz - 3) * coeffi[19] +
                            C44 * (zz * (35 * zz - 30) + 3) * coeffi[20] +
                            C45 * xz * (7 * zz - 3) * coeffi[21] +
                            C46 * (xx - yy) * (7 * zz - 1) * coeffi[22] +
                            C47 * xz * (xx - 3 * yy) * coeffi[23] +
                            C48 * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * coeffi[24];
                }
            }
                      
        }
    }
    return result;
}
// inline __device__ __host__




inline __device__ __host__ 
void positional_encoding(
    const float3 x,
    const int L,
    float* embed_x)
{
    embed_x[0] = x.x;
    embed_x[1] = x.y;
    embed_x[2] = x.z;
    int idx = 3;
    for (int i=0; i<L; i++)
    {
        embed_x[idx] = sinf(powf(2.0f, 1.0f *i) * x.x);
        embed_x[idx+1] = sinf(powf(2.0f, 1.0f *i) * x.y);
        embed_x[idx+2] = sinf(powf(2.0f, 1.0f *i) * x.z);
        embed_x[idx+3] = cosf(powf(2.0f, 1.0f *i) * x.x);
        embed_x[idx+4] = cosf(powf(2.0f, 1.0f *i) * x.y);
        embed_x[idx+5] = cosf(powf(2.0f, 1.0f *i) * x.z);
        idx += 6;
    }
}

inline __device__ __host__ 
void Relu(float* x, int size)
{
    for (int i=0; i<size; i++)
    {
        x[i] = x[i] < 0? 0:x[i];
    }
}

inline __device__ __host__
void leakyRelu(float* x, int size, float negative_slope)
{
    for (int i=0; i<size; i++)
    {
        x[i] = x[i] < 0? x[i]*negative_slope:x[i];
    }
}

inline __device__ __host__
void Sigmoid(float* x, int size)
{
    for (int i=0; i<size; i++)
    {
        x[i] = 1.0f / (1.0f + expf(-1.0f * x[i]));
    }
}

inline __device__ __host__
float3 Sigmoid(float3 x)
{
    x.x = 1.0f / (1.0f + expf(-1.0f * x.x));
    x.y = 1.0f / (1.0f + expf(-1.0f * x.y));
    x.z = 1.0f / (1.0f + expf(-1.0f * x.z));
    return x;
}

inline __device__ __host__
void TanH(float* x, int size)
{
    for (int i=0; i<size; i++)
    {
        float a = expf(x[i]);
        float b = expf(-1.0f * x[i]);
        x[i] = (a - b) / (a + b);
    }
}

inline __device__ __host__
float3 TanH(float3 x)
{
    float a,b;
    a = expf(x.x);
    b = expf(-1.0f * x.x);
    x.x = (a-b) / (a+b);

    a = expf(x.y);
    b = expf(-1.0f * x.y);
    x.y = (a-b) / (a+b);

    a = expf(x.z);
    b = expf(-1.0f * x.z);
    x.z = (a-b) / (a+b);
    return x; 
}

inline __device__ __host__ 
void Linear(const float* weight, const float* bias, 
            const int in_features, const int out_features,
            const float* in_x, float* out_x)
{
    // weight out_features x in_features 
    for (int i=0; i<out_features; i++)
    {
        float out = 0.0f;
        const float* cur_weight = weight + i * in_features;

        for (int j=0; j<in_features; j++)
        {
            out += (in_x[j] * cur_weight[j]);
        }
        out_x[i] = out + bias[i];
    }
}


inline __device__ __host__
void inference_coeffi(
    const float* W1, const float* b1,
    const float* W2, const float* b2,
    const float* W3, const float* b3,
    const float* W4, const float* b4,
    const float* W5, const float* b5,
    const float* W6, const float* b6,
    const int in1, const int in2, const int in3, const int in4, const int in5, const int in6, const int in7,
    float3 x, float3 view_dir, int L, int deg, 
    float* x1, float* x2, float* x3, float* x4, float* x5, float* x6, float* x7,
    float3 &rgb, float &sigma)
{
    /*
    W1  in2 x in1   b1 in2 
    W2  in3 x in2   b2 in3 
    W3  out_features x in3 b3 out_features
    x 3   x1  in1   x2  in2   x3  in3 
    raw out_features
    rgb 3
    sigma 1 
    */
    positional_encoding(x, L, x1);
    Linear(W1, b1, in1, in2, x1, x2);
    leakyRelu(x2, in2, 0.01f);
    Linear(W2, b2, in2, in3, x2, x3);
    leakyRelu(x3, in3, 0.01f);
    Linear(W3, b3, in3, in4, x3, x4);
    leakyRelu(x4, in4, 0.01f);

    Linear(W4, b4, in4, in5, x4, x5);
    leakyRelu(x5, in5, 0.01f);
    Linear(W5, b5, in5, in6, x5, x6);
    leakyRelu(x6, in6, 0.01f);
    Linear(W6, b6, in6, in7, x6, x7);
    
    sigma = x7[in7-1];
    sigma = sigma < 0? 0 : sigma;

    // TanH(x4, in4);
    float3 raw = inference_SH(deg, view_dir, (float3*)x7);

    // return TanH(raw);
    rgb = Sigmoid(raw);
}



inline __device__ __host__
void inference_coeffi(
    const float* W1, const float* b1,
    const float* W2, const float* b2,
    const float* W3, const float* b3,
    const int in1, const int in2, const int in3, const int in4, 
    float3 x, float3 view_dir, int L, int deg, 
    float* x1, float* x2, float* x3, float* x4,
    float3 &rgb, float &sigma)
{
    /*
    W1  in2 x in1   b1 in2 
    W2  in3 x in2   b2 in3 
    W3  out_features x in3 b3 out_features
    x 3   x1  in1   x2  in2   x3  in3 
    raw out_features
    rgb 3
    sigma 1 
    */
    positional_encoding(x, L, x1);
    Linear(W1, b1, in1, in2, x1, x2);
    leakyRelu(x2, in2, 0.01f);
    Linear(W2, b2, in2, in3, x2, x3);
    leakyRelu(x3, in3, 0.01f);
    Linear(W3, b3, in3, in4, x3, x4);

    sigma = x4[in4-1];
    sigma = sigma < 0? 0 : sigma;

    // TanH(x4, in4);
    float3 raw = inference_SH(deg, view_dir, (float3*)x4);

    // return TanH(raw);
    rgb = Sigmoid(raw);
}


inline __device__ __host__
float3 inference_coeffi(
    const float* W1, const float* b1,
    const float* W2, const float* b2,
    const float* W3, const float* b3,
    const int in1, const int in2, const int in3, const int in4, 
    float3 x, float3 view_dir, int L, int deg, 
    float* x1, float* x2, float* x3, float* x4)
{
    /*
    W1  in2 x in1   b1 in2 
    W2  in3 x in2   b2 in3 
    W3  out_features x in3 b3 out_features
    x 3   x1  in1   x2  in2   x3  in3 
    raw out_features
    rgb 3
    sigma 1 
    */
    positional_encoding(x, L, x1);
    Linear(W1, b1, in1, in2, x1, x2);
    leakyRelu(x2, in2, 0.01f);
    Linear(W2, b2, in2, in3, x2, x3);
    leakyRelu(x3, in3, 0.01f);
    Linear(W3, b3, in3, in4, x3, x4);

    // TanH(x4, in4);
    float3 raw = inference_SH(deg, view_dir, (float3*)x4);

    // return TanH(raw);
    return Sigmoid(raw);
}
// struct Linear
// {
//     float *weight, *bias; 
//     int in_channel , out_channel;

//     Linear(){}
//     void init(int in_channel, int out_channel, float* weight, float* bias)
//     {
//         this->in_channel = in_channel;
//         this->out_channel = out_channel;
//         this->weight = weight;
//         this->bias = bias;
//     }
//     void forward(float* x_in, float* x_out)
//     {
//         for (int i=0; i<this->out_channel; i++)
//         {
//             float out = 0.0f;
//             float* cur_weight = this->weight + i * this->in_channel;

//             for (int j=0; j<this->in_channel; j++)
//             {
//                 out += (x_in[j] * cur_weight[j]);
//             }
//             x_out[i] = out + this->bias[i];
//         }
//     }
// };



// struct MLP
// {
//     Linear F1, F2, F3;
//     int L, in1, in2, in3; 
//     float *x1, *x2, *x3, *x4;
//     void init(float* W1, float* b1, 
//               float* W2, float* b2,
//               float* W3, float* b3,
//               float* x1, float* x2, float* x3,
//               int in1, int in2, int in3,
//               int in4, int L, int deg)
//     {
//         this->F1 = Linear(in1, in2, W1, b1);
//         this->F2 = Linear(in2, in3, W2, b2);
//         this->F3 = Linear(in3, in4, W3, b3);
//         this->x1 = x1;
//         this->x2 = x2;
//         this->x3 = x3;
//         this->in1 = in1;
//         this->in2 = in2;
//         this->in3 = in3;
//         this->in4 = in4;
//         this->L = L;
//         this->deg = deg; 
//     }
//     float3 forward(int ray_idx, float3 x, float3 view_dir)
//     {
//         float* x1 = this->x1 + ray_idx*this->in1;
//         float* x2 = this->x2 + ray_idx*this->in2;
//         float* x3 = this->x3 + ray_idx*this->in3;
//         float* x4 = this->x4 + ray_idx*this->in4;

//         positional_encoding(x, this->L, x1);
//         this->F1.forward(x1, x2);
//         leakyRelu(x2, in2, 0.01f);
//         this->F2.forward(x2,x3);
//         leakyRelu(x3, in3, 0.01f);
//         this->F3.forward(x3, x4);

//         // normalized view dir 
//         TanH(x4, this->in4);
//         return inference_SH_nodiffuse(deg, view_dir, x4);
//     }

// };


inline __device__ __host__
void inference(const float* W1, const float* b1,
               const float* W2, const float* b2,
               const float* W3, const float* b3,
               const int in1, const int in2, const int in3,
               const int out_features, float3 x,
               float* x1, float* x2, float* x3, float* raw, 
               float3 &rgb, float &sigma, int L)
{
    /*
    W1  in2 x in1   b1 in2 
    W2  in3 x in2   b2 in3 
    W3  out_features x in3 b3 out_features
    x 3   x1  in1   x2  in2   x3  in3 
    raw out_features
    rgb 3
    sigma 1 
    */
    positional_encoding(x, L, x1);
    Linear(W1, b1, in1, in2, x1, x2);
    Relu(x2, in2);
    Linear(W2, b2, in2, in3, x2, x3);
    Relu(x3, in3);
    Linear(W3, b3, in3, out_features, x3, raw);

    Sigmoid(raw, 3);
    Relu(raw+3,1);

    rgb = make_float3(raw[0], raw[1], raw[2]);
    sigma = raw[3];
}

#endif 