gcc -o bin/vm1 vm1.c \
-I$(conda info --base)/include/python3.12 \
-L$(conda info --base)/lib -lpython3.12

export DYLD_LIBRARY_PATH=$(conda info --base)/lib:$DYLD_LIBRARY_PATH

./vm1 ../pyc/example.cpython-312.pyc
