

```
(base) cccimac@cccimacdeiMac 02 % python frozenLakeImprove.py 
09:35:22 [INFO] observation space = Discrete(16)
09:35:22 [INFO] action space = Discrete(4)
09:35:22 [INFO] number of states = 16
09:35:22 [INFO] number of actions = 4
09:35:22 [INFO] P[14] = {0: [(0.3333333333333333, 10, 0.0, False), (0.3333333333333333, 13, 0.0, False), (0.3333333333333333, 14, 0.0, False)], 1: [(0.3333333333333333, 13, 0.0, False), (0.3333333333333333, 14, 0.0, False), (0.3333333333333333, 15, 1.0, True)], 2: [(0.3333333333333333, 14, 0.0, False), (0.3333333333333333, 15, 1.0, True), (0.3333333333333333, 10, 0.0, False)], 3: [(0.3333333333333333, 15, 1.0, True), (0.3333333333333333, 10, 0.0, False), (0.3333333333333333, 13, 0.0, False)]}
09:35:22 [INFO] P[14][2] = [(0.3333333333333333, 14, 0.0, False), (0.3333333333333333, 15, 1.0, True), (0.3333333333333333, 10, 0.0, False)]
09:35:22 [INFO] reward threshold = 0.7
09:35:22 [INFO] max episode steps = 100
09:35:22 [INFO] ==== Random policy ====
/opt/miniconda3/lib/python3.12/site-packages/gym/utils/passive_env_checker.py:233: DeprecationWarning: `np.bool8` is a deprecated alias for `np.bool_`.  (Deprecated NumPy 1.24)
  if not isinstance(terminated, (bool, np.bool8)):
09:35:22 [INFO] average episode reward = 0.02 ± 0.14
09:35:22 [INFO] state value:
[[0.0139372  0.01162942 0.02095187 0.01047569]
 [0.01624741 0.         0.04075119 0.        ]
 [0.03480561 0.08816967 0.14205297 0.        ]
 [0.         0.17582021 0.43929104 0.        ]]
09:35:22 [INFO] action value:
[[0.01470727 0.01393801 0.01393801 0.01316794]
 [0.00852221 0.01162969 0.01086043 0.01550616]
 [0.02444416 0.0209521  0.02405958 0.01435233]
 [0.01047585 0.01047585 0.00698379 0.01396775]
 [0.02166341 0.01701767 0.0162476  0.01006154]
 [0.         0.         0.         0.        ]
 [0.05433495 0.04735099 0.05433495 0.00698396]
 [0.         0.         0.         0.        ]
 [0.01701767 0.04099176 0.03480569 0.04640756]
 [0.0702086  0.11755959 0.10595772 0.05895286]
 [0.18940397 0.17582024 0.16001408 0.04297362]
 [0.         0.         0.         0.        ]
 [0.         0.         0.         0.        ]
 [0.08799662 0.20503708 0.23442697 0.17582024]
 [0.25238807 0.53837042 0.52711467 0.43929106]
 [0.         0.         0.         0.        ]]
09:35:22 [INFO] Updating completes. Updated policy is:
[[1. 0. 0. 0.]
 [0. 0. 0. 1.]
 [1. 0. 0. 0.]
 [0. 0. 0. 1.]
 [1. 0. 0. 0.]
 [1. 0. 0. 0.]
 [1. 0. 0. 0.]
 [1. 0. 0. 0.]
 [0. 0. 0. 1.]
 [0. 1. 0. 0.]
 [1. 0. 0. 0.]
 [1. 0. 0. 0.]
 [1. 0. 0. 0.]
 [0. 0. 1. 0.]
 [0. 1. 0. 0.]
 [1. 0. 0. 0.]]
09:35:22 [INFO] optimal state value =
[[0.82351246 0.82350689 0.82350303 0.82350106]
 [0.82351416 0.         0.5294002  0.        ]
 [0.82351683 0.82352026 0.76469786 0.        ]
 [0.         0.88234658 0.94117323 0.        ]]
09:35:22 [INFO] optimal policy =
[[0 3 3 3]
 [0 0 0 0]
 [3 1 0 0]
 [0 2 1 0]]
09:35:22 [INFO] average episode reward = 0.79 ± 0.41
09:35:22 [INFO] optimal state value =
[[0.82351232 0.82350671 0.82350281 0.82350083]
 [0.82351404 0.         0.52940011 0.        ]
 [0.82351673 0.82352018 0.76469779 0.        ]
 [0.         0.88234653 0.94117321 0.        ]]
09:35:22 [INFO] optimal policy = 
[[0 3 3 3]
 [0 0 0 0]
 [3 1 0 0]
 [0 2 1 0]]
09:35:22 [INFO] average episode reward = 0.80 ± 0.40
```
