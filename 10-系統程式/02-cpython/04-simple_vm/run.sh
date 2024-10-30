set -x
# gcc
g++ -o simple_vm simple_vm.c \
-I$(conda info --base)/include/python3.12 \
-L$(conda info --base)/lib -lpython3.12

export DYLD_LIBRARY_PATH=$(conda info --base)/lib:$DYLD_LIBRARY_PATH

./simple_vm pyc/example.cpython-312.pyc
