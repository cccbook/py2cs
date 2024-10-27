

```
(base) cccimac@cccimacdeiMac 06-cpyvm % ./build.sh
dyld[3653]: symbol '__ZTINSt3__13pmr15memory_resourceE' missing from root that overrides /usr/lib/libc++.1.dylib. Use of that symbol in /System/Library/PrivateFrameworks/caulk.framework/Versions/A/caulk is being set to 0xBAD4007.
Disassembling code object: <module>
Filename: example.py
First Line Number: 1
Constants:
  [0] Non-string constant
  [1] add(2,3)=
  [2] Non-string constant
  [3] Non-string constant
  [4] Non-string constant
Names:
  [0] add
  [1] print
Variable Names:
  [0] add
  [1] print
Bytecode:
   0: RESUME 0
   2: LOAD_CONST 0
   4: MAKE_FUNCTION 0
   6: STORE_NAME 0
   8: PUSH_NULL 0
  10: LOAD_NAME 1
  12: LOAD_CONST 1
  14: PUSH_NULL 0
  16: LOAD_NAME 0
  18: LOAD_CONST 2
  20: LOAD_CONST 3
  22: CALL 2
  30: CALL 2
  38: POP_TOP 0
  40: RETURN_CONST 4
Disassembling code object: <module>
Filename: example.py
First Line Number: 1
Constants:
   0: code:<code object add at 0x104475070, file "example.py", line 1>
   1: str:add(2,3)= 
   2: int:2 
   3: int:3 
   4: NoneType:None 
Names:
   0: add
   1: print
Bytecode:
   0: RESUME 0   # 
   2: LOAD_CONST 0       # <code object add at 0x104475070, file "example.py", line 1>
   4: MAKE_FUNCTION 0    # 
   6: STORE_NAME 0       # 'add'
   8: PUSH_NULL 0        # 
  10: LOAD_NAME 1        # 'print'
  12: LOAD_CONST 1       # 'add(2,3)='
  14: PUSH_NULL 0        # 
  16: LOAD_NAME 0        # 'add'
  18: LOAD_CONST 2       # 2
  20: LOAD_CONST 3       # 3
  22: CALL 2     # 
  30: CALL 2     # 
  38: POP_TOP 0          # 
  40: RETURN_CONST 4     # None
```
