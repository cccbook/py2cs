# Python 模組

https://realpython.com/python-modules-packages/


The Module Search Path
Continuing with the above example, ’s take a look at what happens when Python executes the statement:

import mod
When the interpreter executes the above import statement, it searches for mod.py in a list of directories assembled from the following sources:

The directory from which the input script was run or the current directory if the interpreter is being run interactively
The list of directories contained in the PYTHONPATH environment variable, if it is set. (The format for PYTHONPATH is OS-dependent but should mimic the PATH environment variable.)
An installation-dependent list of directories configured at the time Python is installed