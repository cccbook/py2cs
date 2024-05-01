
```
$ python cartpole_dump.py
env.action_space= Discrete(2)
env.observation_space= Box([-4.8000002e+00 -3.4028235e+38 -4.1887903e-01 -3.4028235e+38], [4.8000002e+00 3.4028235e+38 4.1887903e-01 3.4028235e+38], (4,), float32)

 observation
 |  | Num | Observation           | Min                 | Max               |

 |  |-----|-----------------------|---------------------|-------------------|

 |  | 0   | Cart Position         | -4.8                | 4.8               |

 |  | 1   | Cart Velocity         | -Inf                | Inf               |

 |  | 2   | Pole Angle            | ~ -0.418 rad (-24°) | ~ 0.418 rad (24°) |

 |  | 3   | Pole Angular Velocity | -Inf                | Inf               |


action= 1
  r= (array([ 0.02727336,  0.18847767,  0.03625453, -0.26141977], dtype=float32), 1.0, False, False, {})
action= 0
  r= (array([ 0.03104291, -0.00714255,  0.03102613,  0.04247424], dtype=float32), 1.0, False, False, {})
```
