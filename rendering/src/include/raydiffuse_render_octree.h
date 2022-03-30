#ifndef _DIFFUSE_RENDERING_OCTREE_H
#define _DIFFUSE_RENDERING_OCTREE_H

#include "predefine.h"

__host__
void rendering_diffuse_octree_fp16(
    at::Tensor rays_d, at::Tensor visitedTiles,
    at::Tensor origin,
    at::Tensor groupMap, 
    at::Tensor data_voxels,
    at::Tensor block_IndexMap, 
    at::Tensor nodes_IndexMap,
    at::Tensor nodes_sampleFlag,
    at::Tensor voxels_start,
    at::Tensor centers,
    float tile_size, float voxel_size,
    float sample_step, int num_voxel, int num_thread,
    float trans_th, float early_stop, bool soft_hit, 
    at::Tensor &frame_diffuse,
    at::Tensor &inverse_near, at::Tensor &netIdxs);


#endif 