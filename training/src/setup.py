from setuptools import setup, find_packages
import unittest,os 
from typing import List

from torch.utils.cpp_extension import BuildExtension, CUDAExtension
print(find_packages())
CUDA_FLAGS = []  # type: List[str]

headers = [os.path.join(os.path.split(os.path.abspath(__file__))[0], 'include')]

ext_modules = [
    CUDAExtension('CUDA_EXT', [
        'Sample.cpp',
        'Sample_kernel.cu',
        'SceneRender.cpp',
        'SceneRender_kernel.cu',
        'Helper.cpp',
        'Helper_kernel.cu',
        'binding.cpp'
    ],
    include_dirs=headers),
]


INSTALL_REQUIREMENTS = ['numpy', 'torch']

setup(
    name='CUDA_EXT',
    description='cuda extension operation',
    author='',
    version='0.1',
    install_requires=INSTALL_REQUIREMENTS,
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension}
)
