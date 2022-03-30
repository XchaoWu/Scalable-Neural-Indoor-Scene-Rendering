#ifndef _COMPUTE_RAYS_H
#define _COMPUTE_RAYS_H

#include "predefine.h"

__host__ 
void compute_rays(
    at::Tensor intrinsic, 
    at::Tensor c2w_rotation,
    int num_thread,
    at::Tensor &rays_d);


#endif 