#ifndef _PREDEFINE_H
#define _PREDEFINE_H

#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <ATen/TensorAccessor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>
#include <ATen/ATen.h>
#include <cassert>
#include <cuda.h>
#include "cuda_fp16.h"
#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/unique.h>
#include "interpolation.h"
#include "cutil_math.h"
#include "cuda_utils.h"

#define CHECK_INPUT(x)                                                         \
  TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor");                   \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")



#endif 