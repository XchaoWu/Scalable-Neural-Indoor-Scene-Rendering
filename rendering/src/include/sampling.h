#ifndef _SAMPLING_H
#define _SAMPLING_H

#include "predefine.h"

__host__ 
void inverse_z_sampling(  
    at::Tensor origin,
    at::Tensor rays_d,
    at::Tensor inverse_near,
    at::Tensor inverse_bound,
    int num_thread,
    int offset, int sample_per_ray,
    at::Tensor &sample_points,
    at::Tensor &integrate_steps);
#endif 