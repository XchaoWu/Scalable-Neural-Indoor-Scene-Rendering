from setuptools import setup, find_packages
import unittest,os 
from typing import List
from glob import glob 
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
print(find_packages())
CUDA_FLAGS = [] 

headers = [os.path.join(os.path.split(os.path.abspath(__file__))[0], 'include'),
           os.path.join(os.path.split(os.path.abspath(__file__))[0], 'imgui'),
           os.path.join(os.path.split(os.path.abspath(__file__))[0], 'imgui/backends')]


ui_src = list(glob('imgui/*.cpp')) + ['imgui/backends/imgui_impl_glfw.cpp',
                                        'imgui/backends/imgui_impl_opengl3.cpp']


ext_modules = [
    CUDAExtension('FASTRENDERING', [
        'compute_rays.cu',
        'raytile_tracing.cu',
        'raydiffuse_render_octree_fp16.cu',
        'reorder.cu',
        'rayreflection_render.cu',
        'sampling.cu',
        'volume_rendering.cu',
        'compute_group_boundary.cu',
        'render_to_screen.cpp',
        'binding.cpp',
    ] + ui_src,
    libraries = ['GL', 'GLU', 'glfw3'],
    extra_compile_args= {'nvcc': ["-Xptxas", "-v"]},
    include_dirs=headers),
]


INSTALL_REQUIREMENTS = ['numpy', 'torch']

setup(
    name='FASTRENDERING',
    description='cuda extension operation',
    author='xchao',
    version='0.1',
    install_requires=INSTALL_REQUIREMENTS,
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension}
)
