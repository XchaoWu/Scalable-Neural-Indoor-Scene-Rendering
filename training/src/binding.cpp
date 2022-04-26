#include "Sample.h"
#include "Helper.h"

PYBIND11_MODULE(CUDA_EXT, m){
    m.doc() = "pybind11 torch extension";
    m.def("Sample_uniform", &Sample_uniform, "");
    m.def("Sample_bg", &Sample_bg, "");
    m.def("Sample_sparse", &Sample_sparse, "");
    m.def("dilate_boundary", &dilate_boundary, "");
    m.def("transparency_statistic", &transparency_statistic, "");
    m.def("Sample_reflection", &Sample_reflection, "");
}