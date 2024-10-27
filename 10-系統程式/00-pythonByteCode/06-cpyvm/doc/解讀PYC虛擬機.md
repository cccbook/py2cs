

## Python 範例

```py
def add(a, b):
    return a + b

print('add(2,3)=', add(2,3))
```

## 組合語言

```
(base) cccimac@cccimacdeiMac py % ./dasm_test.sh
  0           0 RESUME                   0

  1           2 LOAD_CONST               0 (<code object add at 0x10509f910, file "../test/example.py", line 1>)
              4 MAKE_FUNCTION            0                 // 創建一個函數
              6 STORE_NAME               0 (add)           // 這個函數的名稱是 add

  4           8 PUSH_NULL                                  // ?
             10 LOAD_NAME                1 (print)         // 準備呼叫 print
             12 LOAD_CONST               1 ('add(2,3)=')   // 放入 'add(2,3)='
             14 PUSH_NULL                                  // ?
             16 LOAD_NAME                0 (add)           // 
             18 LOAD_CONST               2 (2)
             20 LOAD_CONST               3 (3)
             22 CALL                     2                 // 推入 add(2,3) 的結果
             30 CALL                     2                 // print('add(2,3)=', add(2,3))
             38 POP_TOP                                    // 沒用到 return value，拿掉
             40 RETURN_CONST             4 (None)          // 傳回 None

Disassembly of <code object add at 0x10509f910, file "../test/example.py", line 1>:
  1           0 RESUME                   0       // 這是 add 函數

  2           2 LOAD_FAST                0 (a)   // 推入 a
              4 LOAD_FAST                1 (b)   // 推入 b
              6 BINARY_OP                0 (+)   // 取出 a, b, 算出 a+b 推回
             10 RETURN_VALUE                     // 傳回 a+b
```


## 常數區

```
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
```