import numpy
from distutils.core import setup, Extension
from Cython.Build import cythonize
import os

kdedir = os.path.abspath(os.getcwd())
include_dir = '../include'

setup(
    ext_modules = cythonize("kernel_coefficients.pyx"),
    include_dirs=[numpy.get_include()]
      )

ext_modules=[
    Extension("evaluate",
        sources=["evaluate.pyx"],
        libraries=["m", "kde"],
        library_dirs=[kdedir],
        include_dirs=[include_dir],
        extra_compile_args=["-O3"]
    )
]


setup(
    name="evaluate",
    ext_modules=cythonize(ext_modules),
    include_dirs=[numpy.get_include()],
)

