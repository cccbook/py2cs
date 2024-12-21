

用 ChatGPT 學 BayesNet

* https://chat.openai.com/share/9dc85b46-0d18-4139-9c30-6538fd4e9658

## bayesNet.py

fail

```
$ python bayesNet.py
Traceback (most recent call last):
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\theano\configparser.py", line 238, in fetch_val_for_key
    return self._theano_cfg.get(section, option)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\configparser.py", line 797, in get
    d = self._unify_values(section, vars)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\configparser.py", line 1168, in _unify_values
    raise NoSectionError(section) from None
configparser.NoSectionError: No section: 'blas'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\theano\configparser.py", line 354, in __get__
    val_str = cls.fetch_val_for_key(self.name, delete_key=delete_key)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\theano\configparser.py", line 242, in fetch_val_for_key
    raise KeyError(key)
KeyError: 'blas__ldflags'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "D:\ccc\ccc112a\py2cs\02-演算法\02-方法\09b-蒙地卡羅法\06b-bayesNetChat\bayesNet.py", line 1, in <module>
    import pymc3 as pm
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\pymc3\__init__.py", line 23, in <module>
    import theano
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\theano\__init__.py", line 83, in <module>
    from theano import scalar, tensor
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\theano\tensor\__init__.py", line 20, in <module>
    from theano.tensor import nnet  # used for softmax, sigmoid, etc.
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\theano\tensor\nnet\__init__.py", line 3, in <module>
    from . import opt
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\theano\tensor\nnet\opt.py", line 32, in <module>
    from theano.tensor.nnet.conv import ConvOp, conv2d
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\theano\tensor\nnet\conv.py", line 20, in <module>
    from theano.tensor import blas
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\theano\tensor\blas.py", line 163, in <module>
    from theano.tensor.blas_headers import blas_header_text, blas_header_version
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\theano\tensor\blas_headers.py", line 1016, in <module>
    if not config.blas__ldflags:
           ^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\theano\configparser.py", line 358, in __get__
    val_str = self.default()
              ^^^^^^^^^^^^^^
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\theano\link\c\cmodule.py", line 2621, in default_blas_ldflags
    blas_info = numpy.distutils.__config__.blas_opt_info
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: module 'numpy.distutils.__config__' has no attribute 'blas_opt_info'
```