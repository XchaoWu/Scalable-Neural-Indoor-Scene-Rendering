#ifndef _RAYTILE_TRACING_H
#define _RAYTILE_TRACING_H

#include "predefine.h"


__host__ 
void tracing_tiles(
    at::Tensor rays_d, at::Tensor IndexMap, 
    at::Tensor origin, at::Tensor tile_shape,
    at::Tensor scene_min_corner, at::Tensor scene_size,
    float tile_size, int num_thread,
    at::Tensor &visitedTiles);

#endif 