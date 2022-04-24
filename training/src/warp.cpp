#include <iostream>
#include <Python.h>
#include <thread>

void _compute_matrix(float src_focal, float src_px, float src_py,
                     float src_r11, float src_r12, float src_r13,
                     float src_r21, float src_r22, float src_r23,
                     float src_r31, float src_r32, float src_r33,
                     float src_t1, float src_t2, float src_t3,
                     float dst_focal, float dst_px, float dst_py,
                     float dst_r11, float dst_r12, float dst_r13,
                     float dst_r21, float dst_r22, float dst_r23,
                     float dst_r31, float dst_r32, float dst_r33,
                     float dst_t1, float dst_t2, float dst_t3,
                     float* m)
{
    float dst_focal_inv = 1.0f / dst_focal;

    float dst_c1 = -1.0f * (dst_r11 * dst_t1 + dst_r21 * dst_t2 + dst_r31 * dst_t3);
    float dst_c2 = -1.0f * (dst_r12 * dst_t1 + dst_r22 * dst_t2 + dst_r32 * dst_t3);
    float dst_c3 = -1.0f * (dst_r13 * dst_t1 + dst_r23 * dst_t2 + dst_r33 * dst_t3);

    float p_src[12] = 
    {
        src_focal*src_r11+src_px*src_r31, src_focal*src_r12+src_px*src_r32, src_focal*src_r13+src_px*src_r33, src_focal*src_t1+src_px*src_t3,
        src_focal*src_r21+src_py*src_r31, src_focal*src_r22+src_py*src_r32, src_focal*src_r23+src_py*src_r33, src_focal*src_t2+src_py*src_t3,
        src_r31, src_r32, src_r33, src_t3
    };


    float p_dst_inv[12] = 
    {
        dst_r11*dst_focal_inv, dst_r21*dst_focal_inv, -1.0f*(dst_r11*dst_px+dst_r21*dst_py)*dst_focal_inv+dst_r31, dst_c1,
        dst_r12*dst_focal_inv, dst_r22*dst_focal_inv, -1.0f*(dst_r12*dst_px+dst_r22*dst_py)*dst_focal_inv+dst_r32, dst_c2,
        dst_r13*dst_focal_inv, dst_r23*dst_focal_inv, -1.0f*(dst_r13*dst_px+dst_r23*dst_py)*dst_focal_inv+dst_r33, dst_c3
    };

    m[0] = p_src[0] * p_dst_inv[0] + p_src[1] * p_dst_inv[4] + p_src[2] * p_dst_inv[8];
    m[1] = p_src[0] * p_dst_inv[1] + p_src[1] * p_dst_inv[5] + p_src[2] * p_dst_inv[9];
    m[2] = p_src[0] * p_dst_inv[2] + p_src[1] * p_dst_inv[6] + p_src[2] * p_dst_inv[10];
    m[3] = p_src[0] * p_dst_inv[3] + p_src[1] * p_dst_inv[7] + p_src[2] * p_dst_inv[11] + p_src[3];

    m[4] = p_src[4] * p_dst_inv[0] + p_src[5] * p_dst_inv[4] + p_src[6] * p_dst_inv[8];
    m[5] = p_src[4] * p_dst_inv[1] + p_src[5] * p_dst_inv[5] + p_src[6] * p_dst_inv[9];
    m[6] = p_src[4] * p_dst_inv[2] + p_src[5] * p_dst_inv[6] + p_src[6] * p_dst_inv[10];
    m[7] = p_src[4] * p_dst_inv[3] + p_src[5] * p_dst_inv[7] + p_src[6] * p_dst_inv[11] + p_src[7];

    m[8] = p_src[8] * p_dst_inv[0] + p_src[9] * p_dst_inv[4] + p_src[10] * p_dst_inv[8];
    m[9] = p_src[8] * p_dst_inv[1] + p_src[9] * p_dst_inv[5] + p_src[10] * p_dst_inv[9];
    m[10] = p_src[8] * p_dst_inv[2] + p_src[9] * p_dst_inv[6] + p_src[10] * p_dst_inv[10];
    m[11] = p_src[8] * p_dst_inv[3] + p_src[9] * p_dst_inv[7] + p_src[10] * p_dst_inv[11] + p_src[11];
}

void compute_grid(float* grid, float* m, float* D, int start_i, int start_j, int end_i, int end_j,
                  int height, int width)
{

    int d_index = start_i * width + start_j;
    int index = d_index * 2;

    for (int i=start_i; i<end_i; i++)
    {
        for (int j=start_j; j<end_j; j++)
        {
            float d = D[d_index];
            if (d != 0)
            {
                float x = d * (m[0] * j + m[1] * i + m[2])+ m[3];
                float y = d * (m[4] * j + m[5] * i + m[6]) + m[7];
                float z = d * (m[8] * j + m[9] * i + m[10]) + m[11];
                float inv_z = 1.0f / z;
                grid[index] = x * inv_z;
                grid[index+1] = y * inv_z;
            }else{
                grid[index] = -1.0f;
                grid[index+1] = -1.0f;
            }
            d_index += 1;
            index += 2;
        }
    }
}

PyObject* Pycompute_grid(PyObject* self, PyObject* args)
{
    // data
    /*
    intrinsic + extrinsic 3+9+3 = 15
    two cameras 15 + 15 = 30  float 
    one depth  height*width  float
    two int height width
    */

    char* byte_data;
    int size, height, width;

    if (!PyArg_ParseTuple(args,"s#ii",&byte_data, &size, &height, &width)){
        return NULL;
    }

    float* data = (float*)byte_data;

    float* m = new float[12];

    _compute_matrix(
        data[0],data[1],data[2],
        data[3],data[4],data[5],
        data[6],data[7],data[8],
        data[9],data[10],data[11],
        data[12],data[13],data[14],
        data[15],data[16],data[17],
        data[18],data[19],data[20],
        data[21],data[22],data[23],
        data[24],data[25],data[26],
        data[27],data[28],data[29],
        m
    );

    m[11] += 1e-8f;

    float* D_dst = data + 30; // depth ptr

    int grid_size = height*width*2;

    float* grid = new float[grid_size];

    int num_thread = 4;
    int each_num = height / num_thread;
    int start = 0, end = start + each_num;
    std::thread* T_set = new std::thread[num_thread];
    for (int i=0; i<num_thread; i++)
    {
        T_set[i] = std::thread(compute_grid, grid, m, D_dst, start, 0, end, width, height, width);
        end = end % height;
        start = end;
        end = std::min(end+each_num, height);   
        
    }

    for (int i=0; i<num_thread; i++)
    {
        T_set[i].join();
    }

    delete [] m;

    char* back_data = (char*) grid;
    int back_size = grid_size*4;

    PyObject* Result =  Py_BuildValue("y#", back_data, back_size);

    delete [] back_data;

    return Result;    
}

static PyMethodDef gf_methods[] = {
    {"compute_grid", Pycompute_grid, METH_VARARGS},
    {NULL,NULL}
};

static struct PyModuleDef callModuleDef = {
    PyModuleDef_HEAD_INIT,
    "compute_grid",
    "",
    -1,
    gf_methods
};

extern "C"

PyMODINIT_FUNC PyInit_compute_grid(void){
    PyModule_Create(&callModuleDef);
}

