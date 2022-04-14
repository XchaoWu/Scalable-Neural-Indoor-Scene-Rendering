#ifndef _REFLECTION_RENDERING_H
#define _REFLECTION_RENDERING_H

#include "predefine.h"

__host__ 
void inference_mlp(
    int max_samples_per_ray,
    int num_block, int num_thread,
    float inverse_far, float trans_stop,
    at::Tensor netIdxs,
    at::Tensor transparency,
    at::Tensor query_pixel_indices,
    at::Tensor pixel_strats,
    at::Tensor group_centers,
    at::Tensor samples,
    at::Tensor rays_d,
    at::Tensor params,
    at::Tensor sample_results);
#endif 