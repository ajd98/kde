import numpy
from distutils.core import setup, Extension
from Cython.Build import cythonize
import os

libdirs = [os.path.join(os.path.abspath(os.getcwd()), 'kde', 'cuda'),
           "/opt/cuda/lib64"]
include_dirs = ['./include', './kde/cuda']

ext_modules=[
    Extension("kde.cuda.evaluate",
        sources=["kde/cuda/evaluate.pyx"],
        libraries=["m", "cukde", "cuda", "cudart"],
        library_dirs=libdirs,
        include_dirs=include_dirs,
        extra_compile_args=["-O3"],
        extra_objects=['kde/cuda/libcukde.a'],
        extra_link_args=["-Wall"],
    )
]


setup(
    name="kde.cuda.evaluate",
    ext_modules=cythonize(ext_modules),
    include_dirs=[numpy.get_include()],
)
