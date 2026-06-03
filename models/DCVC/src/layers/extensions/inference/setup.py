# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import glob
import sys
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


import os
nvidia_include = f"{os.environ.get('CONDA_PREFIX', '')}/lib/python3.12/site-packages/nvidia"
extra_include = f"-I{nvidia_include}/cusparse/include -I{nvidia_include}/cublas/include -I{nvidia_include}/cusolver/include -I{nvidia_include}/cuda_runtime/include"
cxx_flags = ["-O3"] + extra_include.split()
nvcc_flags = ["-O3", "--use_fast_math", "--extra-device-vectorization", "-arch=sm_120"] + extra_include.split()
if sys.platform == 'win32':
    cxx_flags = ["/O2"]


setup(
    name='inference_extensions_cuda',
    ext_modules=[
        CUDAExtension(
            name='inference_extensions_cuda',
            sources=glob.glob('*.cpp') + glob.glob('*.cu'),
            extra_compile_args={
                "cxx": cxx_flags,
                "nvcc": nvcc_flags,
            },
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
