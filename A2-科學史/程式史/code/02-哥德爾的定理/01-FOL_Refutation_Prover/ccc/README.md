# FOL_Refutation_Prover

專案來源 -- https://github.com/sameeravithana/FOL_Refutation_Prover

## run

```
$ python prover.py
[0, 1, 4, 8]
```

## 說明

```py
test_cases = [
    ["(FORALL x (IMPLIES (p x) (q x)))", "(p (f a))", "(NOT (q (f a)))"],  # 0 - this is inconsistent
    ["(FORALL x (IMPLIES (p x) (q x)))", "(FORALL x (p x))", "(NOT (FORALL x (q x)))"],  # 1 -this is inconsistent
    ["(EXISTS x (AND (p x) (q b)))", "(FORALL x (p x))"],  # 2 -this should NOT lead to an empty clause
    ["(NOT (NOT (p a)))"],  # 3 - this should NOT lead to an empty clause
    ["(big_f (f a b) (f b c))",
     "(big_f (f b c) (f a c))",
      "(FORALL x (FORALL y (FORALL z (IMPLIES (AND (big_f x y) (big_f y z)) (big_f x z)))))",
      "(NOT (big_f (f a b) (f a c)))"], # 4 - this is inconsistent
    ["(NOT (FORALL x (EXISTS y (AND (IMPLIES (p x) (NOT (NOT (q y)))) (FORALL w (EXISTS u (OR (s w u) (NOT (NOT (t w u))))))))))"], # 5
    ["(FORALL x (IMPLIES (AND (OR (EXISTS y (p y a b c)) (q a b)) (p x y)) (r x)))"], # 6
    ["(FORALL x (EXISTS y (EXISTS z (AND (AND (AND (AND (l x y) (l y z)) (r z)) (IMPLIES (p z) (r z))) (IMPLIES (r z) (p z))))))", "(FORALL x (FORALL y (FORALL z (AND (EXISTS x (FORALL y (NOT (AND (p y) (l x y))))) (IMPLIES (AND (l x y) (l y z)) (l x z))))))"], # 7
    ["(FORALL x (EXISTS y (p x y)))", "(EXISTS x (FORALL y (NOT (p x y))))"], # 8
    ["(FORALL x (OR (NOT (p a)) (q a)))", "(FORALL x (p x))", "(OR (NOT (p (f a))) (NOT (q a)))"], # 9
    ["(FORALL x (FORALL z (FORALL u (FORALL w (OR (p x (f x) z) (p u w w))))))", "(FORALL x (FORALL y (FORALL z (OR (NOT (p x y z)) (NOT (p z z z))))))"] # 10
]
```

執行結果 [0,1,4,8] 代表那幾組邏輯規則是會產生 empty 空集合，也就是你加入的 -Rule 規則會產生矛盾，意思是 Rule 是可以被證明的。

Refutation 是 Robinson 所提出來的一組推論法則，而且是 Refutation Complete 的。

