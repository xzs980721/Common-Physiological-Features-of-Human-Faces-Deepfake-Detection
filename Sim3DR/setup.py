from setuptools import setup, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy as np

ext_modules = [
    Extension(
        "lib.rasterize",
        ["lib/rasterize.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        language='c++',  # 显式指定为 C++
        extra_compile_args=["-std=c++11"]
    ),
    Extension(
        "Sim3DR_Cython",
        sources=["lib/rasterize.pyx", "lib/rasterize_kernel.cpp"],
        language='c++',
        include_dirs=[np.get_include()],
        extra_compile_args=["-std=c++11"]
    )
]

setup(
    name='Sim3DR_Cython',  # not the package name
    cmdclass={'build_ext': build_ext},
    ext_modules=cythonize(ext_modules),
)