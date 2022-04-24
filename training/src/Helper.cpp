#include "Helper.h"


void dilate_boundary_cuda(at::Tensor &nodes_flag, int num_voxel);

void dilate_boundary(at::Tensor &nodes_flag, int num_voxel)
{
    CHECK_INPUT(nodes_flag);

    TORCH_CHECK(nodes_flag.dim() == 3);

    dilate_boundary_cuda(nodes_flag, num_voxel);
}



void transparency_statistic_cuda(
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
    at::Tensor &T_voxels)
{
    CHECK_INPUT(rays_start);
    CHECK_INPUT(rays_dir);
    CHECK_INPUT(tile_center);
    CHECK_INPUT(voxels);
    CHECK_INPUT(nodes);
    CHECK_INPUT(T_voxels);

    TORCH_CHECK(rays_start.dim() == 2);
    TORCH_CHECK(rays_dir.dim() == 2);
    TORCH_CHECK(tile_center.dim() == 1);
    TORCH_CHECK(voxels.dim() == 3);
    TORCH_CHECK(nodes.dim() == 3);
    TORCH_CHECK(T_voxels.dim() == 3);

    transparency_statistic_cuda(rays_start, rays_dir, num_voxel, tile_center, tile_size, voxel_size, sample_step, voxels, nodes, T_voxels);
}
