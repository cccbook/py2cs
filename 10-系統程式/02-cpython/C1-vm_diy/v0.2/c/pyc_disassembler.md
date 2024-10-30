

```
(base) cccimac@cccimacdeiMac 05-cpyvm % ./build.sh
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
```