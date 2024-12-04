from setuptools import setup, Extension

module = Extension('cmatrix', sources=['cmatrix.c'])

setup(
    name='cmatrix',
    version='1.0',
    description='Matrix extension',
    ext_modules=[module],
)
