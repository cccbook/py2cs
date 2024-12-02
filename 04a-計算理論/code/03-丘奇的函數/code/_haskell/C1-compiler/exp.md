

```
GHCi, version 9.4.8: https://www.haskell.org/ghc/  :? for help
ghci> :load exp.hs
[1 of 1] Compiling SimpleCompiler   ( exp.hs, interpreted )
Ok, one module loaded.
ghci> main
輸入表達式:
3+5*8
AST: Mul (Add (IntLit 3) (IntLit 5)) (IntLit 8)
指令: [Push 3,Push 5,AddInstr,Push 8,MulInstr]
結果: 64
ghci> main
輸入表達式:
2 + 3 * 4
AST: IntLit 2
指令: [Push 2]
結果: 2
ghci> 2+3*4
14
ghci> main
輸入表達式:
2+3*4
AST: Mul (Add (IntLit 2) (IntLit 3)) (IntLit 4)
指令: [Push 2,Push 3,AddInstr,Push 4,MulInstr]
結果: 20
```
