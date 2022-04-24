#pragma once

#include "macros.h"
#include <ATen/ATen.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>


void Sample_reflection(
    const at::Tensor rays_start,
    const at::Tensor rays_dir,
    const at::Tensor node_flags,
    const at::Tensor voxels,
    const float cx,
    const float cy,
    const float cz,
    const float tile_size,
    const float far,
    const int num_voxel,
    const float T_th, 
    at::Tensor &z_vals);


void Sample_sparse(
    const at::Tensor rays_start,
    const at::Tensor rays_dir,
    const at::Tensor node_flags,
    const float cx,
    const float cy,
    const float cz,
    const float tile_size,
    const int num_voxel,
    at::Tensor &z_vals,
    at::Tensor &dists);


void Sample_uniform(
    const at::Tensor rays_start,
    const at::Tensor rays_dir,
    const float cx,
    const float cy,
    const float cz,
    const float tile_size,
    at::Tensor &z_vals);


void Sample_bg(
    const at::Tensor rays_start,
    const at::Tensor rays_dir,
    const at::Tensor bg_zval,
    // const at::Tensor last_zval,
    const at::Tensor tile_center,
    const float tile_size,
    // const float voxel_size,
    const float sample_range, 
    at::Tensor &z_vals);