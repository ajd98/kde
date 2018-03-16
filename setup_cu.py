import numpy
from distutils.core import setup, Extension
from Cython.Build import cythonize
import os

libdir = os.path.join(os.path.abspath(os.getcwd()), 'kde', 'cuda')
include_dirs = ['./include', './kde/cuda']

ext_modules=[
    Extension("kde.cuda.evaluate",
        sources=["kde/cuda/evaluate.pyx"],
        libraries=["m", "cukde"],
        library_dirs=[libdir],
        include_dirs=include_dirs,
        extra_compile_args=["-O3"]
    )
]


setup(
    name="kde.cuda.evaluate",
    ext_modules=cythonize(ext_modules),
    include_dirs=[numpy.get_include()],
)
