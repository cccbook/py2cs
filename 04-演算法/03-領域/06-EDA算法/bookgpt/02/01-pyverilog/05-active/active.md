

```
(base) cccimac@cccimacdeiMac 05-active % python active.py -t TOP ../TOP.v
Generating LALR tables
WARNING: 183 shift/reduce conflicts
FSM signal: TOP.count, Condition list length: 4
FSM signal: TOP.state, Condition list length: 5
Condition: (Eq, TOP.enable), Inferring transition condition
Condition: (Ulnot, Eq), Inferring transition condition
Condition: (Ulnot, Ulnot, Eq), Inferring transition condition
```