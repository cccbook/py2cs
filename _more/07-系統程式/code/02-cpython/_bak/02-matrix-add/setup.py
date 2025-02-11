from setuptools import setup, Extension

module = Extension('matrix_add', sources=['matrix_add.c'])

setup(
    name='matrix_add',
    version='1.0',
    description='Matrix addition extension',
    ext_modules=[module],
)
