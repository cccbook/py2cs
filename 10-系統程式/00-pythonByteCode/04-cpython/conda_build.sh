gcc -o pyc_disassembler pyc_disassembler.c \
-I$(conda info --base)/include/python3.12 \
-L$(conda info --base)/lib -lpython3.12

export DYLD_LIBRARY_PATH=$(conda info --base)/lib:$DYLD_LIBRARY_PATH

./pyc_disassembler pyc/example.cpython-312.pyc
