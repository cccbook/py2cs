============================================================
No. 1 Value Iteration
============================================================
[Parameters]
Gamma = 0.99
Threshold = 0.05

[Variables]     
Delta = 1.0     

[State-Value]   
[[ 0.  0. -1. -1.]
============================================================
No. 2 Value Iteration
============================================================
[Parameters]
Gamma = 0.99
Threshold = 0.05

[Variables]
Delta = 0.99

[State-Value]
[[ 0.    0.   -1.   -1.99]
============================================================
No. 3 Value Iteration
============================================================
[Parameters]
Gamma = 0.99
Threshold = 0.05

[Variables]
Delta = 0.0

[State-Value]
[[ 0.    0.   -1.   -1.99]
============================================================
Final Result
============================================================
[State-value]
[[ 0.    0.   -1.   -1.99]
 [ 0.   -1.   -1.99 -1.  ]
 [-1.   -1.99 -1.    0.  ]
 [-1.99 -1.    0.    0.  ]]
============================================================
[Policy]
[['*' '<' '<' '<']
 ['^' '^' '^' 'v']
 ['^' '^' 'v' 'v']
 ['^' '>' '>' '*']]
============================================================