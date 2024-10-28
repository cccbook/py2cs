
gcc -o my_module my_module.c \
-I$(conda info --base)/include/python3.12 \
-L$(conda info --base)/lib -lpython3.12