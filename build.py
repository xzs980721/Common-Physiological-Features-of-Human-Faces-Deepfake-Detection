# coding: utf-8

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import os
from os.path import join as pjoin
import numpy as np
from setuptools import setup, Extension
from Cython.Distutils import build_ext

def find_in_path(name, path):
    "Find a file in a search path"
    # adapted from http://code.activestate.com/recipes/52224-find-a-file-given-a-search-path/
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None

# Obtain the numpy include directory. This logic works across numpy versions.
try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()

# run the customize_compiler
class custom_build_ext(build_ext):
    def build_extensions(self):
        # customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)

ext_modules = [
    Extension(
        "nms.cpu_nms",
        ["FaceBoxes/utils/nms/cpu_nms.pyx"],
        extra_compile_args=["-Wno-cpp", "-Wno-unused-function"],
        include_dirs=[numpy_include]
    )
]

setup(
    name='mot_utils',
    ext_modules=ext_modules,
    cmdclass={'build_ext': custom_build_ext},
)