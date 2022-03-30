#ifndef _VOLUME_RENDERING_H
#define _VOLUME_RENDERING_H

#include "predefine.h"

__host__
void integrate_points(
    at::Tensor sample_results,
    at::Tensor intergrate_steps,
    int num_thread, float early_step, 
    at::Tensor &transparency,
    at::Tensor &colors_buffer);

__host__
void frame_memcpy(long frame_pointer, const at::Tensor &frame);
#endif 