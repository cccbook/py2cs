python py/gen_op_c.py > c/op.c
python py/gen_op_h.py > c/op.h

gcc -o bin/pyc_disassembler c/pyc_disassembler.c c/op.c \
-I$(conda info --base)/include/python3.12 \
-L$(conda info --base)/lib -lpython3.12

export DYLD_LIBRARY_PATH=$(conda info --base)/lib:$DYLD_LIBRARY_PATH

./bin/pyc_disassembler pyc/example.cpython-312.pyc
