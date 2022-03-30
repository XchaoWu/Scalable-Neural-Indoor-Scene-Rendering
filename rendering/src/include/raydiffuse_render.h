#ifndef _DIFFUSE_RENDERING_H
#define _DIFFUSE_RENDERING_H

#include "predefine.h"


__host__
void rendering_diffuse(
    at::Tensor rays_d, at::Tensor visitedTiles,
    at::Tensor SparseToGroup, at::Tensor origin,
    at::Tensor groupMap, at::Tensor voxels,
    at::Tensor nodes, at::Tensor centers,
    float tile_size, float voxel_size,
    float sample_step, int num_voxel, int num_thread,
    float trans_th, 
    at::Tensor &frame_diffuse,
    at::Tensor &inverse_near, at::Tensor &netIdxs);


#endif 