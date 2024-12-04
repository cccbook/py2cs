from setuptools import setup, Extension

matrix_module = Extension('matrix', sources=['matrix.c'])

setup(
    name='matrix',
    version='1.0',
    description='A simple matrix library',
    ext_modules=[matrix_module],
)
