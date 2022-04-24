#pragma once

#include "macros.h"
#include <ATen/ATen.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>


void dilate_boundary(at::Tensor &nodes_flag, int num_voxel);


void transparency_statistic(
    const at::Tensor rays_start,
    const at::Tensor rays_dir,
    const int num_voxel,
    const at::Tensor tile_center,
    const float tile_size,
    const float voxel_size,
    const float sample_step,
    at::Tensor voxels,
    at::Tensor nodes,
    at::Tensor &T_voxels);