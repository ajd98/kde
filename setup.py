import numpy
from distutils.core import setup, Extension
from Cython.Build import cythonize
import os

libdir = os.path.join(os.path.abspath(os.getcwd()), 'kde', 'evaluate')
include_dir = './include'

setup(
    ext_modules = cythonize("kde/evaluate/kernel_coefficients.pyx"),
    include_dirs=[numpy.get_include()]
      )

ext_modules=[
    Extension("kde.evaluate._evaluate",
        sources=["kde/evaluate/_evaluate.pyx"],
        libraries=["m", "kde"],
        library_dirs=[libdir],
        include_dirs=[include_dir],
        extra_compile_args=["-O3"]
    )
]


setup(
    name="kde.evaluate._evaluate",
    ext_modules=cythonize(ext_modules),
    include_dirs=[numpy.get_include()],
)

