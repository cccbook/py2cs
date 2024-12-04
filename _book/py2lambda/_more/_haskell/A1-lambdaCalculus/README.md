

```
(base) cccimac@cccimacdeiMac 03-lambdaCalculus % ghci
GHCi, version 9.4.8: https://www.haskell.org/ghc/  :? for help
ghci> :load lambdaCalculus.hs 
[1 of 1] Compiling LambdaCalculus   ( lambdaCalculus.hs, interpreted )

lambdaCalculus.hs:11:12: error:
    • Couldn't match type ‘b’ with ‘a’
      Expected: a -> b -> a
        Actual: a -> b -> b
      ‘b’ is a rigid type variable bound by
        the type signature for:
          not' :: forall a b c. ((a -> b -> a) -> (a -> b -> b) -> c) -> c
        at lambdaCalculus.hs:10:1-50
      ‘a’ is a rigid type variable bound by
        the type signature for:
          not' :: forall a b c. ((a -> b -> a) -> (a -> b -> b) -> c) -> c
        at lambdaCalculus.hs:10:1-50
    • In the first argument of ‘c’, namely ‘false’
      In the expression: c false true
      In an equation for ‘not'’: not' c = c false true
    • Relevant bindings include
        c :: (a -> b -> a) -> (a -> b -> b) -> c
          (bound at lambdaCalculus.hs:11:6)
        not' :: ((a -> b -> a) -> (a -> b -> b) -> c) -> c
          (bound at lambdaCalculus.hs:11:1)
   |
11 | not' c = c false true
   |            ^^^^^

lambdaCalculus.hs:14:16: error:
    • Couldn't match type ‘a’ with ‘a -> b -> a’
      Expected: a -> b -> a
        Actual: (a -> b -> a) -> (a -> b -> a) -> c
      ‘a’ is a rigid type variable bound by
        the type signature for:
          and' :: forall a b c. ((a -> b -> a) -> (a -> b -> a) -> c) -> c
        at lambdaCalculus.hs:13:1-50
    • In the second argument of ‘p’, namely ‘p’
      In the expression: p q p
      In an equation for ‘and'’: and' p q = p q p
    • Relevant bindings include
        q :: a -> b -> a (bound at lambdaCalculus.hs:14:8)
        p :: (a -> b -> a) -> (a -> b -> a) -> c
          (bound at lambdaCalculus.hs:14:6)
        and' :: ((a -> b -> a) -> (a -> b -> a) -> c) -> c
          (bound at lambdaCalculus.hs:14:1)
   |
14 | and' p q = p q p
   |                ^

lambdaCalculus.hs:17:13: error:
    • Couldn't match type ‘a’ with ‘a -> b -> a’
      Expected: a -> b -> a
        Actual: (a -> b -> a) -> (a -> b -> a) -> c
      ‘a’ is a rigid type variable bound by
        the type signature for:
          or' :: forall a b c. ((a -> b -> a) -> (a -> b -> a) -> c) -> c
        at lambdaCalculus.hs:16:1-49
    • In the first argument of ‘p’, namely ‘p’
      In the expression: p p q
      In an equation for ‘or'’: or' p q = p p q
    • Relevant bindings include
        q :: a -> b -> a (bound at lambdaCalculus.hs:17:7)
        p :: (a -> b -> a) -> (a -> b -> a) -> c
          (bound at lambdaCalculus.hs:17:5)
        or' :: ((a -> b -> a) -> (a -> b -> a) -> c) -> c
          (bound at lambdaCalculus.hs:17:1)
   |
17 | or' p q = p p q
   |             ^

lambdaCalculus.hs:20:23: error:
    • Couldn't match type ‘b’ with ‘a4 -> b4 -> b4’
      Expected: a -> b -> a
        Actual: (a4 -> b4 -> a4) -> (a4 -> b4 -> b4) -> a -> b -> a
      ‘b’ is a rigid type variable bound by
        the type signature for:
          xor' :: forall a b c. ((a -> b -> a) -> (a -> b -> a) -> c) -> c
        at lambdaCalculus.hs:19:1-50
    • In the second argument of ‘p’, namely ‘q’
      In the expression: p (not' q) q
      In an equation for ‘xor'’: xor' p q = p (not' q) q
    • Relevant bindings include
        q :: (a4 -> b4 -> a4) -> (a4 -> b4 -> b4) -> a -> b -> a
          (bound at lambdaCalculus.hs:20:8)
        p :: (a -> b -> a) -> (a -> b -> a) -> c
          (bound at lambdaCalculus.hs:20:6)
        xor' :: ((a -> b -> a) -> (a -> b -> a) -> c) -> c
          (bound at lambdaCalculus.hs:20:1)
   |
20 | xor' p q = p (not' q) q
   |                       ^

lambdaCalculus.hs:24:13: error:
    • Couldn't match expected type ‘c -> d’ with actual type ‘d’
      ‘d’ is a rigid type variable bound by
        the type signature for:
          if' :: forall a b c d.
                 ((a -> b -> a) -> c -> d) -> a -> b -> c -> d
        at lambdaCalculus.hs:23:1-52
    • In the expression: c x y
      In an equation for ‘if'’: if' c x y = c x y
    • Relevant bindings include
        c :: (a -> b -> a) -> c -> d (bound at lambdaCalculus.hs:24:5)
        if' :: ((a -> b -> a) -> c -> d) -> a -> b -> c -> d
          (bound at lambdaCalculus.hs:24:1)
   |
24 | if' c x y = c x y
   |             ^^^^^

lambdaCalculus.hs:34:15: error:
    • Couldn't match expected type ‘(a3 -> a3) -> b’
                  with actual type ‘e’
      ‘e’ is a rigid type variable bound by
        the type signature for:
          pred' :: forall a b c d e f g.
                   ((a -> b -> c) -> d -> e) -> (b -> f) -> g -> b
        at lambdaCalculus.hs:33:1-56
    • The function ‘n’ is applied to three value arguments,
        but its type ‘(a -> b -> c) -> d -> e’ has only two
      In the expression: n (\ g h -> h (g f)) (const x) id
      In an equation for ‘pred'’:
          pred' n f x = n (\ g h -> h (g f)) (const x) id
    • Relevant bindings include
        f :: b -> f (bound at lambdaCalculus.hs:34:9)
        n :: (a -> b -> c) -> d -> e (bound at lambdaCalculus.hs:34:7)
        pred' :: ((a -> b -> c) -> d -> e) -> (b -> f) -> g -> b
          (bound at lambdaCalculus.hs:34:1)
   |
34 | pred' n f x = n (\g h -> h (g f)) (const x) id
   |               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

lambdaCalculus.hs:34:26: error:
    • Couldn't match expected type ‘t0 -> c’ with actual type ‘b’
      ‘b’ is a rigid type variable bound by
        the type signature for:
          pred' :: forall a b c d e f g.
                   ((a -> b -> c) -> d -> e) -> (b -> f) -> g -> b
        at lambdaCalculus.hs:33:1-56
    • The function ‘h’ is applied to one value argument,
        but its type ‘b’ has none
      In the expression: h (g f)
      In the first argument of ‘n’, namely ‘(\ g h -> h (g f))’
    • Relevant bindings include
        h :: b (bound at lambdaCalculus.hs:34:21)
        f :: b -> f (bound at lambdaCalculus.hs:34:9)
        n :: (a -> b -> c) -> d -> e (bound at lambdaCalculus.hs:34:7)
        pred' :: ((a -> b -> c) -> d -> e) -> (b -> f) -> g -> b
          (bound at lambdaCalculus.hs:34:1)
   |
34 | pred' n f x = n (\g h -> h (g f)) (const x) id
   |                          ^^^^^^^

lambdaCalculus.hs:34:36: error:
    • Couldn't match expected type ‘d’ with actual type ‘b2 -> g’
      ‘d’ is a rigid type variable bound by
        the type signature for:
          pred' :: forall a b c d e f g.
                   ((a -> b -> c) -> d -> e) -> (b -> f) -> g -> b
        at lambdaCalculus.hs:33:1-56
    • In the second argument of ‘n’, namely ‘(const x)’
      In the expression: n (\ g h -> h (g f)) (const x) id
      In an equation for ‘pred'’:
          pred' n f x = n (\ g h -> h (g f)) (const x) id
    • Relevant bindings include
        x :: g (bound at lambdaCalculus.hs:34:11)
        n :: (a -> b -> c) -> d -> e (bound at lambdaCalculus.hs:34:7)
        pred' :: ((a -> b -> c) -> d -> e) -> (b -> f) -> g -> b
          (bound at lambdaCalculus.hs:34:1)
   |
34 | pred' n f x = n (\g h -> h (g f)) (const x) id
   |                                    ^^^^^^^

lambdaCalculus.hs:38:20: error:
    • Couldn't match expected type ‘c’ with actual type ‘a’
      ‘a’ is a rigid type variable bound by
        the type signature for:
          add :: forall a b c.
                 ((a -> b) -> c -> a) -> ((a -> b) -> c -> a) -> (a -> b) -> c -> a
        at lambdaCalculus.hs:37:1-73
      ‘c’ is a rigid type variable bound by
        the type signature for:
          add :: forall a b c.
                 ((a -> b) -> c -> a) -> ((a -> b) -> c -> a) -> (a -> b) -> c -> a
        at lambdaCalculus.hs:37:1-73
    • In the second argument of ‘m’, namely ‘(n f x)’
      In the expression: m f (n f x)
      In an equation for ‘add’: add m n f x = m f (n f x)
    • Relevant bindings include
        x :: c (bound at lambdaCalculus.hs:38:11)
        f :: a -> b (bound at lambdaCalculus.hs:38:9)
        n :: (a -> b) -> c -> a (bound at lambdaCalculus.hs:38:7)
        m :: (a -> b) -> c -> a (bound at lambdaCalculus.hs:38:5)
        add :: ((a -> b) -> c -> a)
               -> ((a -> b) -> c -> a) -> (a -> b) -> c -> a
          (bound at lambdaCalculus.hs:38:1)
   |
38 | add m n f x = m f (n f x)
   |                    ^^^^^

lambdaCalculus.hs:41:18: error:
    • Couldn't match type ‘a’ with ‘b3 -> b3 -> c0’
      Expected: (a -> b) -> c -> a
        Actual: ((b3 -> b3 -> c0) -> g0 -> b3) -> c -> a
      ‘a’ is a rigid type variable bound by
        the type signature for:
          sub :: forall a b c.
                 ((a -> b) -> c -> a) -> ((a -> b) -> c -> a) -> (a -> b) -> c -> a
        at lambdaCalculus.hs:40:1-73
    • In the second argument of ‘add’, namely ‘(neg n)’
      In the expression: add m (neg n)
      In an equation for ‘sub’:
          sub m n
            = add m (neg n)
            where
                neg n f x = n (pred' f) x
    • Relevant bindings include
        n :: (a -> b) -> c -> a (bound at lambdaCalculus.hs:41:7)
        m :: (a -> b) -> c -> a (bound at lambdaCalculus.hs:41:5)
        sub :: ((a -> b) -> c -> a)
               -> ((a -> b) -> c -> a) -> (a -> b) -> c -> a
          (bound at lambdaCalculus.hs:41:1)
   |
41 | sub m n = add m (neg n)
   |                  ^^^^^

lambdaCalculus.hs:45:17: error:
    • Couldn't match type ‘c’ with ‘a’
      Expected: a -> b
        Actual: c -> a
      ‘c’ is a rigid type variable bound by
        the type signature for:
          mult :: forall a b c.
                  ((a -> b) -> c -> a) -> ((a -> b) -> c -> a) -> (a -> b) -> c -> a
        at lambdaCalculus.hs:44:1-74
      ‘a’ is a rigid type variable bound by
        the type signature for:
          mult :: forall a b c.
                  ((a -> b) -> c -> a) -> ((a -> b) -> c -> a) -> (a -> b) -> c -> a
        at lambdaCalculus.hs:44:1-74
    • In the first argument of ‘m’, namely ‘(n f)’
      In the expression: m (n f)
      In an equation for ‘mult’: mult m n f = m (n f)
    • Relevant bindings include
        f :: a -> b (bound at lambdaCalculus.hs:45:10)
        n :: (a -> b) -> c -> a (bound at lambdaCalculus.hs:45:8)
        m :: (a -> b) -> c -> a (bound at lambdaCalculus.hs:45:6)
        mult :: ((a -> b) -> c -> a)
                -> ((a -> b) -> c -> a) -> (a -> b) -> c -> a
          (bound at lambdaCalculus.hs:45:1)
   |
45 | mult m n f = m (n f)
   |                 ^^^

lambdaCalculus.hs:48:11: error:
    • Couldn't match type ‘c’ with ‘a’
      Expected: a -> b -> c
        Actual: c -> a
      ‘c’ is a rigid type variable bound by
        the type signature for:
          pow :: forall a b c.
                 ((a -> b) -> c -> a) -> ((a -> b) -> c -> a) -> a -> b -> c
        at lambdaCalculus.hs:47:1-66
      ‘a’ is a rigid type variable bound by
        the type signature for:
          pow :: forall a b c.
                 ((a -> b) -> c -> a) -> ((a -> b) -> c -> a) -> a -> b -> c
        at lambdaCalculus.hs:47:1-66
    • In the expression: y x
      In an equation for ‘pow’: pow x y = y x
    • Relevant bindings include
        y :: (a -> b) -> c -> a (bound at lambdaCalculus.hs:48:7)
        x :: (a -> b) -> c -> a (bound at lambdaCalculus.hs:48:5)
        pow :: ((a -> b) -> c -> a) -> ((a -> b) -> c -> a) -> a -> b -> c
          (bound at lambdaCalculus.hs:48:1)
   |
48 | pow x y = y x
   |           ^^^

lambdaCalculus.hs:64:11: error:
    • Couldn't match expected type ‘Int’ with actual type ‘a’
      ‘a’ is a rigid type variable bound by
        the type signature for:
          toInt :: forall a b c. ((a -> b) -> c -> a) -> Int
        at lambdaCalculus.hs:63:1-36
    • In the expression: n (+ 1) 0
      In an equation for ‘toInt’: toInt n = n (+ 1) 0
    • Relevant bindings include
        n :: (a -> b) -> c -> a (bound at lambdaCalculus.hs:64:7)
        toInt :: ((a -> b) -> c -> a) -> Int
          (bound at lambdaCalculus.hs:64:1)
   |
64 | toInt n = n (+1) 0
   |           ^^^^^^^^

lambdaCalculus.hs:67:13: error:
    • Couldn't match type ‘b’ with ‘c’
      Expected: (a -> b) -> c -> a
        Actual: (a -> b) -> b -> b
      ‘b’ is a rigid type variable bound by
        the type signature for:
          fromInt :: forall a b c. Int -> (a -> b) -> c -> a
        at lambdaCalculus.hs:66:1-38
      ‘c’ is a rigid type variable bound by
        the type signature for:
          fromInt :: forall a b c. Int -> (a -> b) -> c -> a
        at lambdaCalculus.hs:66:1-38
    • In the expression: zero
      In an equation for ‘fromInt’: fromInt 0 = zero
    • Relevant bindings include
        fromInt :: Int -> (a -> b) -> c -> a
          (bound at lambdaCalculus.hs:67:1)
   |
67 | fromInt 0 = zero
   |             ^^^^
Failed, no modules loaded.
ghci> :load lambdaCalculus.hs 
[1 of 1] Compiling LambdaCalculus   ( lambdaCalculus.hs, interpreted )

lambdaCalculus.hs:43:22: error:
    • Couldn't match type ‘a4’ with ‘a3 -> a3’
      Expected: ((a3 -> a3) -> a3 -> a3) -> a4 -> a4
        Actual: ((a3 -> a3) -> a3 -> a3) -> (a3 -> a3) -> a3 -> a3
      ‘a4’ is a rigid type variable bound by
        a type expected by the context:
          ChurchNum
        at lambdaCalculus.hs:43:17-23
    • In the first argument of ‘neg’, namely ‘n’
      In the second argument of ‘add’, namely ‘(neg n)’
      In the expression: add m (neg n)
   |
43 | sub m n = add m (neg n)
   |                      ^

lambdaCalculus.hs:45:26: error:
    • Couldn't match expected type ‘(a5 -> a5) -> a5 -> a5’
                  with actual type ‘p’
      ‘p’ is a rigid type variable bound by
        the inferred type of
          neg :: (((a4 -> a4) -> a4 -> a4) -> t1 -> t2) -> p -> t1 -> t2
        at lambdaCalculus.hs:45:5-29
    • In the first argument of ‘pred'’, namely ‘f’
      In the first argument of ‘n’, namely ‘(pred' f)’
      In the expression: n (pred' f) x
    • Relevant bindings include
        f :: p (bound at lambdaCalculus.hs:45:11)
        neg :: (((a4 -> a4) -> a4 -> a4) -> t1 -> t2) -> p -> t1 -> t2
          (bound at lambdaCalculus.hs:45:5)
   |
45 |     neg n f x = n (pred' f) x
   |                          ^

lambdaCalculus.hs:78:33: error:
    • No instance for (Eq (a0 -> a0 -> a0)) arising from a use of ‘==’
        (maybe you haven't applied a function to enough arguments?)
    • In the second argument of ‘test’, namely ‘(not' true == false)’
      In a stmt of a 'do' block: test "Not works" (not' true == false)
      In the expression:
        do test "Successor works" (toInt (succ' zero) == 1)
           test "Addition works" (toInt (add (fromInt 2) (fromInt 3)) == 5)
           test
             "Multiplication works" (toInt (mult (fromInt 2) (fromInt 3)) == 6)
           test "Power works" (toInt (pow (fromInt 2) (fromInt 3)) == 8)
           ....
   |
78 |     test "Not works" (not' true == false)
   |                                 ^^

lambdaCalculus.hs:79:38: error:
    • No instance for (Eq (a1 -> a1 -> a1)) arising from a use of ‘==’
        (maybe you haven't applied a function to enough arguments?)
    • In the second argument of ‘test’, namely
        ‘(and' true true == true)’
      In a stmt of a 'do' block:
        test "And works" (and' true true == true)
      In the expression:
        do test "Successor works" (toInt (succ' zero) == 1)
           test "Addition works" (toInt (add (fromInt 2) (fromInt 3)) == 5)
           test
             "Multiplication works" (toInt (mult (fromInt 2) (fromInt 3)) == 6)
           test "Power works" (toInt (pow (fromInt 2) (fromInt 3)) == 8)
           ....
   |
79 |     test "And works" (and' true true == true)
   |                                      ^^

lambdaCalculus.hs:80:37: error:
    • No instance for (Eq (a2 -> a2 -> a2)) arising from a use of ‘==’
        (maybe you haven't applied a function to enough arguments?)
    • In the second argument of ‘test’, namely
        ‘(or' false true == true)’
      In a stmt of a 'do' block: test "Or works" (or' false true == true)
      In the expression:
        do test "Successor works" (toInt (succ' zero) == 1)
           test "Addition works" (toInt (add (fromInt 2) (fromInt 3)) == 5)
           test
             "Multiplication works" (toInt (mult (fromInt 2) (fromInt 3)) == 6)
           test "Power works" (toInt (pow (fromInt 2) (fromInt 3)) == 8)
           ....
   |
80 |     test "Or works" (or' false true == true)
   |                                     ^^
Failed, no modules loaded.
```
