

```sh
(base) cccimac@cccimacdeiMac 00-basic % ghci
GHCi, version 9.4.8: https://www.haskell.org/ghc/  :? for help
ghci> :load add.hs
[1 of 2] Compiling Main             ( add.hs, interpreted )
Ok, one module loaded.
ghci> main
8
ghci> :load map.hs
[1 of 2] Compiling Main             ( map.hs, interpreted )
Ok, one module loaded.
ghci> main
[1,4,9,16]
ghci> :load lazy.hs
[1 of 2] Compiling Main             ( lazy.hs, interpreted )
Ok, one module loaded.
ghci> main

<interactive>:6:1: error:
    Variable not in scope: main
    Suggested fix: Perhaps use ‘min’ (imported from Prelude)
ghci> :load lazy.hs
[1 of 2] Compiling Main             ( lazy.hs, interpreted )
Ok, one module loaded.
ghci> main
[1,2,3,4,5,6,7,8,9,10]
ghci> :load struct.hs
[1 of 2] Compiling Main             ( struct.hs, interpreted )
Ok, one module loaded.
ghci> main
78.53982
24.0
ghci> :load monad.hs
[1 of 2] Compiling Main             ( monad.hs, interpreted )
Ok, one module loaded.
ghci> main
*** Exception: example.txt: openFile: does not exist (No such file or directory)
ghci> main
this is example.txt

ghci> :load quicksort.hs
[1 of 2] Compiling Main             ( quicksort.hs, interpreted )
Ok, one module loaded.
ghci> main
[1,1,3,4,5,9]
ghci> :load compose.hs
[1 of 2] Compiling Main             ( compose.hs, interpreted )
Ok, one module loaded.
ghci> main
[4,8,12]
ghci> :load fib.hs
[1 of 2] Compiling Main             ( fib.hs, interpreted )
Ok, one module loaded.
ghci> main
[0,1,1,2,3,5,8,13,21,34]
```