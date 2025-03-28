

```
(base) cccimac@cccimacdeiMac 04-controlflow-fail % python controlflow.py -t TOP ../TOP.v
FSM signal: TOP.count, Condition list length: 4
FSM signal: TOP.state, Condition list length: 5
Condition: (Eq, TOP.enable), Inferring transition condition
Condition: (Ulnot, Eq), Inferring transition condition
Condition: (Ulnot, Ulnot, Eq), Inferring transition condition
# SIGNAL NAME: TOP.state
# DELAY CNT: 0
0 --(TOP_enable>'d0)--> 1
1 --None--> 2
2 --None--> 0
Traceback (most recent call last):
  File "/Users/cccimac/Desktop/ccc/py2cs_bak/04-演算法/03-領域/06-EDA算法/bookgpt/02/01-pyverilog/04-controlflow/controlflow.py", line 93, in <module>
    main()
  File "/Users/cccimac/Desktop/ccc/py2cs_bak/04-演算法/03-領域/06-EDA算法/bookgpt/02/01-pyverilog/04-controlflow/controlflow.py", line 84, in main
    fsm.tograph(filename=util.toFlatname(signame) + '.' +
  File "/opt/miniconda3/lib/python3.12/site-packages/pyverilog/controlflow/controlflow_analyzer.py", line 228, in tograph
    import pygraphviz as pgv
ModuleNotFoundError: No module named 'pygraphviz'
```
