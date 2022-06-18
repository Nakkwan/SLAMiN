#!/usr/bin/env python3
import os
import torch

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

cxx_args = ['-std=c++11']

nvcc_args = [
    #'-gencode', 'arch=compute_75,code=sm_75',
    #'-gencode', 'arch=compute_75,code=sm_75',
    #'-gencode', 'arch=compute_86,code=sm_86',
    #'-gencode', 'arch=compute_86,code=sm_86',
    '-gencode', 'arch=compute_70,code=sm_70',
    '-gencode', 'arch=compute_70,code=compute_70'
]

setup(
    name='resample2d_cuda',
    ext_modules=[
        CUDAExtension('resample2d_cuda', [
            'resample2d_cuda.cc',
            'resample2d_kernel.cu'
        ], extra_compile_args={'cxx': cxx_args, 'nvcc': nvcc_args})
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

