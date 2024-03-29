from setuptools import setup
from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy



setup(
    ext_modules=cythonize("test_solution_wc.pyx"),
    include_dirs=[numpy.get_include()]
)    