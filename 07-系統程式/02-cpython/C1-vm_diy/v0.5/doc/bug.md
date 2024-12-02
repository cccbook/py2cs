

help: Type help() for interactive help, or help(object) for help about object.
vm start...
   0: RESUME 0   #  待處理 ...
   2: LOAD_CONST 0       # <code object add at 0x104d29070, file "example.py", line 1>
   4: MAKE_FUNCTION 0    # <code object add at 0x104d29070, file "example.py", line 1>
   6: STORE_NAME 0       # <function add at 0x104832340>
   8: PUSH_NULL 0        # 
  10: LOAD_NAME 1        # <built-in function print>
  12: LOAD_CONST 1       # 'add(2,3)='
  14: PUSH_NULL 0        # 
  16: LOAD_NAME 0        # 這裡會有問題，因為 globals 裡沒有 add
  18: LOAD_CONST 2       # 2
  20: LOAD_CONST 3       # 3
./build.sh: line 10:  5237 Segmentation fault: 11  ./bin/pyc_vm pyc/example.cpython-312.pyc
(base) cccimac@cccimacdeiMac v0.5 % 