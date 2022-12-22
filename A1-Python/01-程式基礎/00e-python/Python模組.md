## Python模組

* https://docs.python.org/3/tutorial/modules.html

模組間的引用

```
from . import echo
from .. import formats
from ..filters import equalizer
```

## 發布套件

* 套件結構請參考 -- https://github.com/ccc-py/nn6

打包前請務必先更新 setup.py 的版本號，然後做下列動作：

```
$ python setup.py sdist bdist_wheel
$ twine upload dist/*
```

範例:

```
PS D:\ccc\code\python\nn6> python setup.py sdist bdist_wheel
running sdist
running egg_info
writing nn6.egg-info\PKG-INFO
writing dependency_links to nn6.egg-info\dependency_links.txt
writing top-level names to nn6.egg-info\top_level.txt
reading manifest file 'nn6.egg-info\SOURCES.txt'
writing manifest file 'nn6.egg-info\SOURCES.txt'
running check
creating nn6-0.0.4
creating nn6-0.0.4\nn6
creating nn6-0.0.4\nn6.egg-info
copying files to nn6-0.0.4...
copying README.md -> nn6-0.0.4
copying setup.py -> nn6-0.0.4
copying nn6\__init__.py -> nn6-0.0.4\nn6
copying nn6.egg-info\PKG-INFO -> nn6-0.0.4\nn6.egg-info
copying nn6.egg-info\SOURCES.txt -> nn6-0.0.4\nn6.egg-info
copying nn6.egg-info\dependency_links.txt -> nn6-0.0.4\nn6.egg-info
copying nn6.egg-info\top_level.txt -> nn6-0.0.4\nn6.egg-info
Writing nn6-0.0.4\setup.cfg
Creating tar archive
removing 'nn6-0.0.4' (and everything under it)
running bdist_wheel
running build
running build_py
installing to build\bdist.win-amd64\wheel
running install
running install_lib
creating build\bdist.win-amd64\wheel
creating build\bdist.win-amd64\wheel\nn6
copying build\lib\nn6\nn.py -> build\bdist.win-amd64\wheel\.\nn6
copying build\lib\nn6\__init__.py -> build\bdist.win-amd64\wheel\.\nn6
running install_egg_info
Copying nn6.egg-info to build\bdist.win-amd64\wheel\.\nn6-0.0.4-py3.7.egg-info
running install_scripts
adding license file "LICENSE.txt" (matched pattern "LICEN[CS]E*")
creating build\bdist.win-amd64\wheel\nn6-0.0.4.dist-info\WHEEL
creating 'dist\nn6-0.0.4-py3-none-any.whl' and adding 'build\bdist.win-amd64\wheel' to it
adding 'nn6/__init__.py'
adding 'nn6/nn.py'
adding 'nn6-0.0.4.dist-info/LICENSE.txt'
adding 'nn6-0.0.4.dist-info/METADATA'
adding 'nn6-0.0.4.dist-info/WHEEL'
adding 'nn6-0.0.4.dist-info/top_level.txt'
adding 'nn6-0.0.4.dist-info/RECORD'
removing build\bdist.win-amd64\wheel
PS D:\ccc\code\python\nn6> twine upload dist/nn6-0.0.4-*
Uploading distributions to https://upload.pypi.org/legacy/
Enter your username: ccckmit
Enter your password:
Uploading nn6-0.0.4-py3-none-any.whl
100%|███████████████████████████████████████████████
100%|███████████████████████████████████████████████
██████████| 6.53k/6.53k [00:04<00:00, 1.60kB/s]

View at:
https://pypi.org/project/nn6/0.0.4/
```

## 與 node.js 整合

* [Integrating Node and Python](https://medium.com/@HolmesLaurence/integrating-node-and-python-6b8454bfc272)


<!--
* [How to upload your python package to PyPi](https://medium.com/@joel.barmettler/how-to-upload-your-python-package-to-pypi-65edc5fe9c56)
    * https://github.com/joelbarmettlerUZH/Scrapeasy
* https://packaging.python.org/tutorials/packaging-projects/
    * https://github.com/pypa/sampleproject

發布方法

```
python setup.py sdist
// 接著請到 github 按下 release, 並將新 release 網址放入 setup.py
twine upload dist/*
```

發布範例

```
PS D:\ccc\code\python\ai6> python setup.py sdist
running sdist
running egg_info
writing ai6.egg-info\PKG-INFO
writing dependency_links to ai6.egg-info\dependency_links.txt
writing requirements to ai6.egg-info\requires.txt
writing top-level names to ai6.egg-info\top_level.txt
package init file 'ai6\__init__.py' not found (or not a regular file)
reading manifest file 'ai6.egg-info\SOURCES.txt'
reading manifest template 'MANIFEST.in'
warning: no files found matching '*' under directory 'data'
writing manifest file 'ai6.egg-info\SOURCES.txt'
running check
creating ai6-1.0.3
creating ai6-1.0.3\ai6
creating ai6-1.0.3\ai6.egg-info
copying files to ai6-1.0.3...
copying LICENSE.txt -> ai6-1.0.3
copying MANIFEST.in -> ai6-1.0.3
copying README.md -> ai6-1.0.3
copying pyproject.toml -> ai6-1.0.3
copying setup.cfg -> ai6-1.0.3
copying setup.py -> ai6-1.0.3
copying ai6\nn.py -> ai6-1.0.3\ai6
copying ai6.egg-info\PKG-INFO -> ai6-1.0.3\ai6.egg-info
copying ai6.egg-info\SOURCES.txt -> ai6-1.0.3\ai6.egg-info
copying ai6.egg-info\dependency_links.txt -> ai6-1.0.3\ai6.egg-info
copying ai6.egg-info\requires.txt -> ai6-1.0.3\ai6.egg-info
copying ai6.egg-info\top_level.txt -> ai6-1.0.3\ai6.egg-info
Writing ai6-1.0.3\setup.cfg
Creating tar archive
removing 'ai6-1.0.3' (and everything under it)

// 接著請到 github 按下 release, 並將新網址

PS D:\ccc\code\python\ai6> twine upload dist/*
Uploading distributions to https://upload.pypi.org/legacy/
Enter your username: ccckmit
Enter your password:
Uploading ai6-0.11.tar.gz
100%|██████████████████████████████████████
100%|██████████████████████████████████████
██| 5.25k/5.25k [00:06<00:00, 881B/s]
Uploading ai6-0.12.tar.gz
100%|██████████████████████████████████████
| 6.17k/6.17k [00:01<00:00, 6.31kB/s]
Uploading ai6-1.0.0.tar.gz
100%|██████████████████████████████████████
| 9.85k/9.85k [00:01<00:00, 8.70kB/s]
Uploading ai6-1.0.1.tar.gz
100%|██████████████████████████████████████
| 9.95k/9.95k [00:01<00:00, 9.72kB/s]
Uploading ai6-1.0.2.tar.gz
100%|██████████████████████████████████████
| 7.17k/7.17k [00:01<00:00, 5.90kB/s]
Uploading ai6-1.0.3.tar.gz
100%|██████████████████████████████████████
| 7.17k/7.17k [00:01<00:00, 4.55kB/s]

View at:
https://pypi.org/project/ai6/1.0.2/
https://pypi.org/project/ai6/0.11/
https://pypi.org/project/ai6/1.0.0/
https://pypi.org/project/ai6/1.0.3/
https://pypi.org/project/ai6/0.12/
https://pypi.org/project/ai6/1.0.1/
```

使用範例

```
$ pip install ai6 -U

$ python useAi6.py
p= [1.0, 3.0] f(p)= 10.0 glen= 6.337207586942211
```
-->