#ifndef _REORDER_H
#define _REORDER_H

#include "predefine.h"

__host__
int sort_by_key(at::Tensor &keys_tensor, at::Tensor &values_tensor, at::Tensor &starts_tensor);

__host__
void assign_pixels(at::Tensor netIdxs, // n-1
                   at::Tensor pixel_starts, // n 
                   at::Tensor block_starts, // n
                   int num_thread,
                   at::Tensor &new_netIdxs,
                   at::Tensor &new_pixel_starts);
#endif 