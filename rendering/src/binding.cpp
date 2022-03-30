#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "compute_rays.h"
#include "raytile_tracing.h"
#include "raydiffuse_render_octree.h"
#include "reorder.h"
#include "rayreflection_render.h"
#include "compute_group_boundary.h"
#include "sampling.h"
#include "volume_rendering.h"
#include "render_to_screen.h"

PYBIND11_MODULE(FASTRENDERING, m)
{
    m.doc() = "pybind11 torch extension";
    m.def("compute_rays", &compute_rays, "");
    m.def("tracing_tiles", &tracing_tiles, "");
    m.def("rendering_diffuse_octree_fp16", &rendering_diffuse_octree_fp16, "");
    m.def("sort_by_key", &sort_by_key, "");
    m.def("assign_pixels", &assign_pixels, "");
    m.def("inference_mlp", &inference_mlp, "");
    m.def("inverse_z_sampling", &inverse_z_sampling, "");
    m.def("integrate_points", &integrate_points, "");
    m.def("compute_group_boundary", &compute_group_boundary, "");
    m.def("render_to_screen", &render_to_screen, "");
    m.def("frame_memcpy", &frame_memcpy, "");
}