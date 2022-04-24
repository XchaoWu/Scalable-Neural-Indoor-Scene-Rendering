#include "Sample.h"

void Sample_sparse_cuda(
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


void Sample_uniform_cuda(
    const at::Tensor rays_start,
    const at::Tensor rays_dir,
    const float cx,
    const float cy,
    const float cz,
    const float tile_size,
    at::Tensor &z_vals);

void Sample_bg_cuda(
    const at::Tensor rays_start,
    const at::Tensor rays_dir,
    const at::Tensor bg_zval,
    // const at::Tensor last_zval,
    const at::Tensor tile_center,
    const float tile_size,
    // const float voxel_size,
    const float sample_range, 
    at::Tensor &z_vals);

void Sample_reflection_cuda(
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
    at::Tensor &z_vals)
{
    CHECK_INPUT(rays_start);
    CHECK_INPUT(rays_dir);
    CHECK_INPUT(z_vals);
    CHECK_INPUT(node_flags);
    CHECK_INPUT(voxels);

    TORCH_CHECK(rays_start.dim() == 2);
    TORCH_CHECK(rays_dir.dim() == 2);
    TORCH_CHECK(z_vals.dim() == 2);
    TORCH_CHECK(node_flags.dim() == 3);
    TORCH_CHECK(voxels.dim() == 4);

    Sample_reflection_cuda(rays_start, rays_dir, node_flags, voxels,
                       cx, cy, cz, tile_size, far, num_voxel, T_th,
                       z_vals);
}

void Sample_bg(
    const at::Tensor rays_start,
    const at::Tensor rays_dir,
    const at::Tensor bg_zval,
    // const at::Tensor last_zval,
    const at::Tensor tile_center,
    const float tile_size,
    // const float voxel_size,
    const float sample_range, 
    at::Tensor &z_vals)
{
    CHECK_INPUT(rays_start);
    CHECK_INPUT(rays_dir);
    CHECK_INPUT(tile_center);
    CHECK_INPUT(bg_zval);
    CHECK_INPUT(z_vals);

    TORCH_CHECK(rays_start.dim() == 2);
    TORCH_CHECK(rays_dir.dim() == 2);
    TORCH_CHECK(tile_center.dim() == 1);
    TORCH_CHECK(bg_zval.dim() == 2);
    TORCH_CHECK(z_vals.dim() == 2);

    Sample_bg_cuda(rays_start, rays_dir, bg_zval, tile_center, tile_size, sample_range, z_vals);
}



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
    at::Tensor &dists)
{
    CHECK_INPUT(rays_start);
    CHECK_INPUT(rays_dir);
    CHECK_INPUT(z_vals);
    CHECK_INPUT(node_flags);
    CHECK_INPUT(dists);

    TORCH_CHECK(rays_start.dim() == 2);
    TORCH_CHECK(rays_dir.dim() == 2);
    TORCH_CHECK(z_vals.dim() == 2);
    TORCH_CHECK(node_flags.dim() == 3);
    TORCH_CHECK(dists.dim() == 2);

    Sample_sparse_cuda(rays_start, rays_dir, node_flags,
                       cx, cy, cz, tile_size, num_voxel,
                       z_vals, dists);
}


void Sample_uniform(
    const at::Tensor rays_start,
    const at::Tensor rays_dir,
    const float cx,
    const float cy,
    const float cz,
    const float tile_size,
    at::Tensor &z_vals) // B x 2 
{
    CHECK_INPUT(rays_start);
    CHECK_INPUT(rays_dir);
    CHECK_INPUT(z_vals);

    TORCH_CHECK(rays_start.dim() == 2);
    TORCH_CHECK(rays_dir.dim() == 2);
    TORCH_CHECK(z_vals.dim() == 2);

    Sample_uniform_cuda(rays_start, rays_dir, cx, cy, cz, tile_size, z_vals);    

}
