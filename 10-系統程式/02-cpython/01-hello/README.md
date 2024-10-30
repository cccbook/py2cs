# CPython  的 Hello 程式

```
(base) cccimac@cccimacdeiMac 01-hello % ./build.sh
running build
running build_ext
running install
/opt/miniconda3/lib/python3.12/site-packages/setuptools/_distutils/cmd.py:66: SetuptoolsDeprecationWarning: setup.py install is deprecated.
!!

        ********************************************************************************
        Please avoid running ``setup.py`` directly.
        Instead, use pypa/build, pypa/installer or other
        standards-based tools.

        See https://blog.ganssle.io/articles/2021/10/setup-py-deprecated.html for details.
        ********************************************************************************

!!
  self.initialize_options()
/opt/miniconda3/lib/python3.12/site-packages/setuptools/_distutils/cmd.py:66: EasyInstallDeprecationWarning: easy_install command is deprecated.
!!

        ********************************************************************************
        Please avoid running ``setup.py`` and ``easy_install``.
        Instead, use pypa/build, pypa/installer or other
        standards-based tools.

        See https://github.com/pypa/setuptools/issues/917 for details.
        ********************************************************************************

!!
  self.initialize_options()
running bdist_egg
running egg_info
writing hello.egg-info/PKG-INFO
writing dependency_links to hello.egg-info/dependency_links.txt
writing top-level names to hello.egg-info/top_level.txt
reading manifest file 'hello.egg-info/SOURCES.txt'
writing manifest file 'hello.egg-info/SOURCES.txt'
installing library code to build/bdist.macosx-11.1-arm64/egg
running install_lib
running build_ext
creating build/bdist.macosx-11.1-arm64/egg
copying build/lib.macosx-11.1-arm64-cpython-312/hello.cpython-312-darwin.so -> build/bdist.macosx-11.1-arm64/egg
creating stub loader for hello.cpython-312-darwin.so
byte-compiling build/bdist.macosx-11.1-arm64/egg/hello.py to hello.cpython-312.pyc
creating build/bdist.macosx-11.1-arm64/egg/EGG-INFO
copying hello.egg-info/PKG-INFO -> build/bdist.macosx-11.1-arm64/egg/EGG-INFO
copying hello.egg-info/SOURCES.txt -> build/bdist.macosx-11.1-arm64/egg/EGG-INFO
copying hello.egg-info/dependency_links.txt -> build/bdist.macosx-11.1-arm64/egg/EGG-INFO
copying hello.egg-info/top_level.txt -> build/bdist.macosx-11.1-arm64/egg/EGG-INFO
writing build/bdist.macosx-11.1-arm64/egg/EGG-INFO/native_libs.txt
zip_safe flag not set; analyzing archive contents...
__pycache__.hello.cpython-312: module references __file__
creating 'dist/hello-1.0-py3.12-macosx-11.1-arm64.egg' and adding 'build/bdist.macosx-11.1-arm64/egg' to it
removing 'build/bdist.macosx-11.1-arm64/egg' (and everything under it)
Processing hello-1.0-py3.12-macosx-11.1-arm64.egg
removing '/opt/miniconda3/lib/python3.12/site-packages/hello-1.0-py3.12-macosx-11.1-arm64.egg' (and everything under it)
creating /opt/miniconda3/lib/python3.12/site-packages/hello-1.0-py3.12-macosx-11.1-arm64.egg
Extracting hello-1.0-py3.12-macosx-11.1-arm64.egg to /opt/miniconda3/lib/python3.12/site-packages
Adding hello 1.0 to easy-install.pth file

Installed /opt/miniconda3/lib/python3.12/site-packages/hello-1.0-py3.12-macosx-11.1-arm64.egg
Processing dependencies for hello==1.0
Finished processing dependencies for hello==1.0
Hello, World!
```
