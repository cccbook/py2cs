

```
(base) cccimac@cccimacdeiMac 04-pyeda % pip install pyeda
DEPRECATION: Loading egg at /opt/miniconda3/lib/python3.12/site-packages/cmatrix-1.0-py3.12-macosx-11.1-arm64.egg is deprecated. pip 24.3 will enforce this behaviour change. A possible replacement is to use pip for package installation. Discussion can be found at https://github.com/pypa/pip/issues/12330
DEPRECATION: Loading egg at /opt/miniconda3/lib/python3.12/site-packages/matrix-1.0-py3.12-macosx-11.1-arm64.egg is deprecated. pip 24.3 will enforce this behaviour change. A possible replacement is to use pip for package installation. Discussion can be found at https://github.com/pypa/pip/issues/12330
DEPRECATION: Loading egg at /opt/miniconda3/lib/python3.12/site-packages/matrix_add-1.0-py3.12-macosx-11.1-arm64.egg is deprecated. pip 24.3 will enforce this behaviour change. A possible replacement is to use pip for package installation. Discussion can be found at https://github.com/pypa/pip/issues/12330
DEPRECATION: Loading egg at /opt/miniconda3/lib/python3.12/site-packages/hello-1.0-py3.12-macosx-11.1-arm64.egg is deprecated. pip 24.3 will enforce this behaviour change. A possible replacement is to use pip for package installation. Discussion can be found at https://github.com/pypa/pip/issues/12330
DEPRECATION: Loading egg at /opt/miniconda3/lib/python3.12/site-packages/matrix-1.0.0-py3.12.egg is deprecated. pip 24.3 will enforce this behaviour change. A possible replacement is to use pip for package installation. Discussion can be found at https://github.com/pypa/pip/issues/12330
Collecting pyeda
  Downloading pyeda-0.29.0.tar.gz (486 kB)
  Preparing metadata (setup.py) ... done
Building wheels for collected packages: pyeda
  Building wheel for pyeda (setup.py) ... error
  error: subprocess-exited-with-error
  
  × python setup.py bdist_wheel did not run successfully.
  │ exit code: 1
  ╰─> [64 lines of output]
      running bdist_wheel
      running build
      running build_py
      creating build/lib.macosx-11.0-arm64-cpython-312/pyeda
      copying pyeda/util.py -> build/lib.macosx-11.0-arm64-cpython-312/pyeda
      copying pyeda/__init__.py -> build/lib.macosx-11.0-arm64-cpython-312/pyeda
      copying pyeda/inter.py -> build/lib.macosx-11.0-arm64-cpython-312/pyeda
      creating build/lib.macosx-11.0-arm64-cpython-312/pyeda/boolalg
      copying pyeda/boolalg/__init__.py -> build/lib.macosx-11.0-arm64-cpython-312/pyeda/boolalg
      copying pyeda/boolalg/bfarray.py -> build/lib.macosx-11.0-arm64-cpython-312/pyeda/boolalg
      copying pyeda/boolalg/boolfunc.py -> build/lib.macosx-11.0-arm64-cpython-312/pyeda/boolalg
      copying pyeda/boolalg/minimization.py -> build/lib.macosx-11.0-arm64-cpython-312/pyeda/boolalg
      copying pyeda/boolalg/table.py -> build/lib.macosx-11.0-arm64-cpython-312/pyeda/boolalg
      copying pyeda/boolalg/bdd.py -> build/lib.macosx-11.0-arm64-cpython-312/pyeda/boolalg
      copying pyeda/boolalg/expr.py -> build/lib.macosx-11.0-arm64-cpython-312/pyeda/boolalg
      creating build/lib.macosx-11.0-arm64-cpython-312/pyeda/logic
      copying pyeda/logic/aes.py -> build/lib.macosx-11.0-arm64-cpython-312/pyeda/logic
      copying pyeda/logic/graycode.py -> build/lib.macosx-11.0-arm64-cpython-312/pyeda/logic
      copying pyeda/logic/__init__.py -> build/lib.macosx-11.0-arm64-cpython-312/pyeda/logic
      copying pyeda/logic/addition.py -> build/lib.macosx-11.0-arm64-cpython-312/pyeda/logic
      copying pyeda/logic/sudoku.py -> build/lib.macosx-11.0-arm64-cpython-312/pyeda/logic
      creating build/lib.macosx-11.0-arm64-cpython-312/pyeda/parsing
      copying pyeda/parsing/lex.py -> build/lib.macosx-11.0-arm64-cpython-312/pyeda/parsing
      copying pyeda/parsing/token.py -> build/lib.macosx-11.0-arm64-cpython-312/pyeda/parsing
      copying pyeda/parsing/pla.py -> build/lib.macosx-11.0-arm64-cpython-312/pyeda/parsing
      copying pyeda/parsing/__init__.py -> build/lib.macosx-11.0-arm64-cpython-312/pyeda/parsing
      copying pyeda/parsing/dimacs.py -> build/lib.macosx-11.0-arm64-cpython-312/pyeda/parsing
      copying pyeda/parsing/boolexpr.py -> build/lib.macosx-11.0-arm64-cpython-312/pyeda/parsing
      creating build/lib.macosx-11.0-arm64-cpython-312/pyeda/test
      copying pyeda/test/__init__.py -> build/lib.macosx-11.0-arm64-cpython-312/pyeda/test
      creating build/lib.macosx-11.0-arm64-cpython-312/pyeda/boolalg/test
      copying pyeda/boolalg/test/test_espresso.py -> build/lib.macosx-11.0-arm64-cpython-312/pyeda/boolalg/test
      copying pyeda/boolalg/test/test_table.py -> build/lib.macosx-11.0-arm64-cpython-312/pyeda/boolalg/test
      copying pyeda/boolalg/test/__init__.py -> build/lib.macosx-11.0-arm64-cpython-312/pyeda/boolalg/test
      copying pyeda/boolalg/test/test_bdd.py -> build/lib.macosx-11.0-arm64-cpython-312/pyeda/boolalg/test
      copying pyeda/boolalg/test/test_bfarray.py -> build/lib.macosx-11.0-arm64-cpython-312/pyeda/boolalg/test
      copying pyeda/boolalg/test/test_expr.py -> build/lib.macosx-11.0-arm64-cpython-312/pyeda/boolalg/test
      copying pyeda/boolalg/test/test_picosat.py -> build/lib.macosx-11.0-arm64-cpython-312/pyeda/boolalg/test
      copying pyeda/boolalg/test/test_exxpr.py -> build/lib.macosx-11.0-arm64-cpython-312/pyeda/boolalg/test
      copying pyeda/boolalg/test/test_boolfunc.py -> build/lib.macosx-11.0-arm64-cpython-312/pyeda/boolalg/test
      creating build/lib.macosx-11.0-arm64-cpython-312/pyeda/logic/test
      copying pyeda/logic/test/test_graycode.py -> build/lib.macosx-11.0-arm64-cpython-312/pyeda/logic/test
      copying pyeda/logic/test/test_addition.py -> build/lib.macosx-11.0-arm64-cpython-312/pyeda/logic/test
      copying pyeda/logic/test/test_sudoku.py -> build/lib.macosx-11.0-arm64-cpython-312/pyeda/logic/test
      copying pyeda/logic/test/__init__.py -> build/lib.macosx-11.0-arm64-cpython-312/pyeda/logic/test
      creating build/lib.macosx-11.0-arm64-cpython-312/pyeda/parsing/test
      copying pyeda/parsing/test/test_pla.py -> build/lib.macosx-11.0-arm64-cpython-312/pyeda/parsing/test
      copying pyeda/parsing/test/test_dimacs.py -> build/lib.macosx-11.0-arm64-cpython-312/pyeda/parsing/test
      copying pyeda/parsing/test/__init__.py -> build/lib.macosx-11.0-arm64-cpython-312/pyeda/parsing/test
      copying pyeda/parsing/test/test_boolexpr.py -> build/lib.macosx-11.0-arm64-cpython-312/pyeda/parsing/test
      running build_ext
      building 'pyeda.boolalg.espresso' extension
      creating build/temp.macosx-11.0-arm64-cpython-312/pyeda/boolalg
      creating build/temp.macosx-11.0-arm64-cpython-312/thirdparty/espresso/src
      clang -fno-strict-overflow -DNDEBUG -O2 -Wall -fPIC -O2 -isystem /opt/miniconda3/include -arch arm64 -fPIC -O2 -isystem /opt/miniconda3/include -arch arm64 -I/opt/homebrew/opt/openjdk/include -Ithirdparty/espresso/src -I/opt/miniconda3/include/python3.12 -c pyeda/boolalg/espressomodule.c -o build/temp.macosx-11.0-arm64-cpython-312/pyeda/boolalg/espressomodule.o
      clang -fno-strict-overflow -DNDEBUG -O2 -Wall -fPIC -O2 -isystem /opt/miniconda3/include -arch arm64 -fPIC -O2 -isystem /opt/miniconda3/include -arch arm64 -I/opt/homebrew/opt/openjdk/include -Ithirdparty/espresso/src -I/opt/miniconda3/include/python3.12 -c thirdparty/espresso/src/cofactor.c -o build/temp.macosx-11.0-arm64-cpython-312/thirdparty/espresso/src/cofactor.o
      thirdparty/espresso/src/cofactor.c:351:50: error: incompatible function pointer types passing 'int (set **, set **)' (aka 'int (unsigned int **, unsigned int **)') to parameter of type 'int (* _Nonnull)(const void *, const void *)' [-Wincompatible-function-pointer-types]
          qsort((char *) (T+2), ncubes, sizeof(set *), d1_order);
                                                       ^~~~~~~~
      /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/stdlib.h:161:22: note: passing argument to parameter '__compar' here
                  int (* _Nonnull __compar)(const void *, const void *));
                                  ^
      1 error generated.
      error: command '/usr/bin/clang' failed with exit code 1
      [end of output]
  
  note: This error originates from a subprocess, and is likely not a problem with pip.
  ERROR: Failed building wheel for pyeda
  Running setup.py clean for pyeda
Failed to build pyeda
ERROR: ERROR: Failed to build installable wheels for some pyproject.toml based projects (pyeda)
```