

```
(base) cccimac@cccimacdeiMac 03-dataflow % python dataflow.py -t TOP ../TOP.v
Directive:
Instance:
(TOP, 'TOP')
Term:
(Term name:TOP.CLK type:['Input'] msb:(IntConst 0) lsb:(IntConst 0))
(Term name:TOP.RST type:['Input'] msb:(IntConst 0) lsb:(IntConst 0))
(Term name:TOP.count type:['Reg'] msb:(IntConst 31) lsb:(IntConst 0))
(Term name:TOP.enable type:['Input'] msb:(IntConst 0) lsb:(IntConst 0))
(Term name:TOP.led type:['Output'] msb:(IntConst 7) lsb:(IntConst 0))
(Term name:TOP.state type:['Reg'] msb:(IntConst 7) lsb:(IntConst 0))
(Term name:TOP.value type:['Input'] msb:(IntConst 31) lsb:(IntConst 0))
Bind:
(Bind dest:TOP.count tree:(Branch Cond:(Terminal TOP.RST) True:(IntConst 0) False:(Branch Cond:(Operator Eq Next:(Terminal TOP.state),(IntConst 0)) False:(Branch Cond:(Operator Eq Next:(Terminal TOP.state),(IntConst 1)) False:(Branch Cond:(Operator Eq Next:(Terminal TOP.state),(IntConst 2)) True:(Operator Plus Next:(Terminal TOP.count),(Terminal TOP.value)))))))
(Bind dest:TOP.led tree:(Partselect Var:(Terminal TOP.count) MSB:(IntConst 23) LSB:(IntConst 16)))
(Bind dest:TOP.state tree:(Branch Cond:(Terminal TOP.RST) True:(IntConst 0) False:(Branch Cond:(Operator Eq Next:(Terminal TOP.state),(IntConst 0)) True:(Branch Cond:(Terminal TOP.enable) True:(IntConst 1)) False:(Branch Cond:(Operator Eq Next:(Terminal TOP.state),(IntConst 1)) True:(IntConst 2) False:(Branch Cond:(Operator Eq Next:(Terminal TOP.state),(IntConst 2)) True:(IntConst 0))))))
```
