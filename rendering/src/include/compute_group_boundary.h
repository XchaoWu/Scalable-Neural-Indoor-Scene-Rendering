#ifndef _COMPUTE_GROUPS_H
#define _COMPUTE_GROUPS_H

#include "predefine.h"


__host__
void compute_group_boundary(
    at::Tensor netIdx, 
    at::Tensor &boundary,
    int ksize,
    int num_thread);

#endif 