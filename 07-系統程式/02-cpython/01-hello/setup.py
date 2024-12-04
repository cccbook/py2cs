from setuptools import setup, Extension

# 定義擴展模組
hello_module = Extension('hello', sources=['hello.c'])

# 設置模組
setup(
    name='hello',
    version='1.0',
    description='A simple Hello World module',
    ext_modules=[hello_module],
)
