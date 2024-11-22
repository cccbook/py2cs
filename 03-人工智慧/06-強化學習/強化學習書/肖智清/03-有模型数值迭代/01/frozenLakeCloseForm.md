

```
(base) cccimac@cccimacdeiMac 03-有模型数值迭代 % python frozenLake1.py 
09:24:23 [INFO] id: FrozenLake-v1
09:24:23 [INFO] entry_point: gym.envs.toy_text.frozen_lake:FrozenLakeEnv
09:24:23 [INFO] reward_threshold: 0.7
09:24:23 [INFO] nondeterministic: False
09:24:23 [INFO] max_episode_steps: 100
09:24:23 [INFO] order_enforce: True
09:24:23 [INFO] autoreset: False
09:24:23 [INFO] disable_env_checker: False
09:24:23 [INFO] apply_api_compatibility: False
09:24:23 [INFO] kwargs: {'map_name': '4x4'}
09:24:23 [INFO] namespace: None
09:24:23 [INFO] name: FrozenLake
09:24:23 [INFO] version: 1
09:24:23 [INFO] desc: [[b'S' b'F' b'F' b'F']
 [b'F' b'H' b'F' b'H']
 [b'F' b'F' b'F' b'H']
 [b'H' b'F' b'F' b'G']]
09:24:23 [INFO] nrow: 4
09:24:23 [INFO] ncol: 4
09:24:23 [INFO] reward_range: (0, 1)
09:24:23 [INFO] initial_state_distrib: [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
09:24:23 [INFO] P: {0: {0: [(0.3333333333333333, 0, 0.0, False), (0.3333333333333333, 0, 0.0, False), (0.3333333333333333, 4, 0.0, False)], 1: [(0.3333333333333333, 0, 0.0, False), (0.3333333333333333, 4, 0.0, False), (0.3333333333333333, 1, 0.0, False)], 2: [(0.3333333333333333, 4, 0.0, False), (0.3333333333333333, 1, 0.0, False), (0.3333333333333333, 0, 0.0, False)], 3: [(0.3333333333333333, 1, 0.0, False), (0.3333333333333333, 0, 0.0, False), (0.3333333333333333, 0, 0.0, False)]}, 1: {0: [(0.3333333333333333, 1, 0.0, False), (0.3333333333333333, 0, 0.0, False), (0.3333333333333333, 5, 0.0, True)], 1: [(0.3333333333333333, 0, 0.0, False), (0.3333333333333333, 5, 0.0, True), (0.3333333333333333, 2, 0.0, False)], 2: [(0.3333333333333333, 5, 0.0, True), (0.3333333333333333, 2, 0.0, False), (0.3333333333333333, 1, 0.0, False)], 3: [(0.3333333333333333, 2, 0.0, False), (0.3333333333333333, 1, 0.0, False), (0.3333333333333333, 0, 0.0, False)]}, 2: {0: [(0.3333333333333333, 2, 0.0, False), (0.3333333333333333, 1, 0.0, False), (0.3333333333333333, 6, 0.0, False)], 1: [(0.3333333333333333, 1, 0.0, False), (0.3333333333333333, 6, 0.0, False), (0.3333333333333333, 3, 0.0, False)], 2: [(0.3333333333333333, 6, 0.0, False), (0.3333333333333333, 3, 0.0, False), (0.3333333333333333, 2, 0.0, False)], 3: [(0.3333333333333333, 3, 0.0, False), (0.3333333333333333, 2, 0.0, False), (0.3333333333333333, 1, 0.0, False)]}, 3: {0: [(0.3333333333333333, 3, 0.0, False), (0.3333333333333333, 2, 0.0, False), (0.3333333333333333, 7, 0.0, True)], 1: [(0.3333333333333333, 2, 0.0, False), (0.3333333333333333, 7, 0.0, True), (0.3333333333333333, 3, 0.0, False)], 2: [(0.3333333333333333, 7, 0.0, True), (0.3333333333333333, 3, 0.0, False), (0.3333333333333333, 3, 0.0, False)], 3: [(0.3333333333333333, 3, 0.0, False), (0.3333333333333333, 3, 0.0, False), (0.3333333333333333, 2, 0.0, False)]}, 4: {0: [(0.3333333333333333, 0, 0.0, False), (0.3333333333333333, 4, 0.0, False), (0.3333333333333333, 8, 0.0, False)], 1: [(0.3333333333333333, 4, 0.0, False), (0.3333333333333333, 8, 0.0, False), (0.3333333333333333, 5, 0.0, True)], 2: [(0.3333333333333333, 8, 0.0, False), (0.3333333333333333, 5, 0.0, True), (0.3333333333333333, 0, 0.0, False)], 3: [(0.3333333333333333, 5, 0.0, True), (0.3333333333333333, 0, 0.0, False), (0.3333333333333333, 4, 0.0, False)]}, 5: {0: [(1.0, 5, 0, True)], 1: [(1.0, 5, 0, True)], 2: [(1.0, 5, 0, True)], 3: [(1.0, 5, 0, True)]}, 6: {0: [(0.3333333333333333, 2, 0.0, False), (0.3333333333333333, 5, 0.0, True), (0.3333333333333333, 10, 0.0, False)], 1: [(0.3333333333333333, 5, 0.0, True), (0.3333333333333333, 10, 0.0, False), (0.3333333333333333, 7, 0.0, True)], 2: [(0.3333333333333333, 10, 0.0, False), (0.3333333333333333, 7, 0.0, True), (0.3333333333333333, 2, 0.0, False)], 3: [(0.3333333333333333, 7, 0.0, True), (0.3333333333333333, 2, 0.0, False), (0.3333333333333333, 5, 0.0, True)]}, 7: {0: [(1.0, 7, 0, True)], 1: [(1.0, 7, 0, True)], 2: [(1.0, 7, 0, True)], 3: [(1.0, 7, 0, True)]}, 8: {0: [(0.3333333333333333, 4, 0.0, False), (0.3333333333333333, 8, 0.0, False), (0.3333333333333333, 12, 0.0, True)], 1: [(0.3333333333333333, 8, 0.0, False), (0.3333333333333333, 12, 0.0, True), (0.3333333333333333, 9, 0.0, False)], 2: [(0.3333333333333333, 12, 0.0, True), (0.3333333333333333, 9, 0.0, False), (0.3333333333333333, 4, 0.0, False)], 3: [(0.3333333333333333, 9, 0.0, False), (0.3333333333333333, 4, 0.0, False), (0.3333333333333333, 8, 0.0, False)]}, 9: {0: [(0.3333333333333333, 5, 0.0, True), (0.3333333333333333, 8, 0.0, False), (0.3333333333333333, 13, 0.0, False)], 1: [(0.3333333333333333, 8, 0.0, False), (0.3333333333333333, 13, 0.0, False), (0.3333333333333333, 10, 0.0, False)], 2: [(0.3333333333333333, 13, 0.0, False), (0.3333333333333333, 10, 0.0, False), (0.3333333333333333, 5, 0.0, True)], 3: [(0.3333333333333333, 10, 0.0, False), (0.3333333333333333, 5, 0.0, True), (0.3333333333333333, 8, 0.0, False)]}, 10: {0: [(0.3333333333333333, 6, 0.0, False), (0.3333333333333333, 9, 0.0, False), (0.3333333333333333, 14, 0.0, False)], 1: [(0.3333333333333333, 9, 0.0, False), (0.3333333333333333, 14, 0.0, False), (0.3333333333333333, 11, 0.0, True)], 2: [(0.3333333333333333, 14, 0.0, False), (0.3333333333333333, 11, 0.0, True), (0.3333333333333333, 6, 0.0, False)], 3: [(0.3333333333333333, 11, 0.0, True), (0.3333333333333333, 6, 0.0, False), (0.3333333333333333, 9, 0.0, False)]}, 11: {0: [(1.0, 11, 0, True)], 1: [(1.0, 11, 0, True)], 2: [(1.0, 11, 0, True)], 3: [(1.0, 11, 0, True)]}, 12: {0: [(1.0, 12, 0, True)], 1: [(1.0, 12, 0, True)], 2: [(1.0, 12, 0, True)], 3: [(1.0, 12, 0, True)]}, 13: {0: [(0.3333333333333333, 9, 0.0, False), (0.3333333333333333, 12, 0.0, True), (0.3333333333333333, 13, 0.0, False)], 1: [(0.3333333333333333, 12, 0.0, True), (0.3333333333333333, 13, 0.0, False), (0.3333333333333333, 14, 0.0, False)], 2: [(0.3333333333333333, 13, 0.0, False), (0.3333333333333333, 14, 0.0, False), (0.3333333333333333, 9, 0.0, False)], 3: [(0.3333333333333333, 14, 0.0, False), (0.3333333333333333, 9, 0.0, False), (0.3333333333333333, 12, 0.0, True)]}, 14: {0: [(0.3333333333333333, 10, 0.0, False), (0.3333333333333333, 13, 0.0, False), (0.3333333333333333, 14, 0.0, False)], 1: [(0.3333333333333333, 13, 0.0, False), (0.3333333333333333, 14, 0.0, False), (0.3333333333333333, 15, 1.0, True)], 2: [(0.3333333333333333, 14, 0.0, False), (0.3333333333333333, 15, 1.0, True), (0.3333333333333333, 10, 0.0, False)], 3: [(0.3333333333333333, 15, 1.0, True), (0.3333333333333333, 10, 0.0, False), (0.3333333333333333, 13, 0.0, False)]}, 15: {0: [(1.0, 15, 0, True)], 1: [(1.0, 15, 0, True)], 2: [(1.0, 15, 0, True)], 3: [(1.0, 15, 0, True)]}}
09:24:23 [INFO] observation_space: Discrete(16)
09:24:23 [INFO] action_space: Discrete(4)
09:24:23 [INFO] render_mode: None
09:24:23 [INFO] window_size: (256, 256)
09:24:23 [INFO] cell_size: (64, 64)
09:24:23 [INFO] window_surface: None
09:24:23 [INFO] clock: None
09:24:23 [INFO] hole_img: None
09:24:23 [INFO] cracked_hole_img: None
09:24:23 [INFO] ice_img: None
09:24:23 [INFO] elf_images: None
09:24:23 [INFO] goal_img: None
09:24:23 [INFO] start_img: None
09:24:23 [INFO] spec: EnvSpec(id='FrozenLake-v1', entry_point='gym.envs.toy_text.frozen_lake:FrozenLakeEnv', reward_threshold=0.7, nondeterministic=False, max_episode_steps=100, order_enforce=True, autoreset=False, disable_env_checker=False, apply_api_compatibility=False, kwargs={'map_name': '4x4'}, namespace=None, name='FrozenLake', version=1)
09:24:23 [INFO] ==== test ====
/opt/miniconda3/lib/python3.12/site-packages/gym/utils/passive_env_checker.py:233: DeprecationWarning: `np.bool8` is a deprecated alias for `np.bool_`.  (Deprecated NumPy 1.24)
  if not isinstance(terminated, (bool, np.bool8)):
09:24:23 [INFO] test episode 0: reward = 1.00, steps = 13
09:24:23 [INFO] test episode 1: reward = 0.00, steps = 10
09:24:23 [INFO] test episode 2: reward = 1.00, steps = 39
09:24:23 [INFO] test episode 3: reward = 1.00, steps = 45
09:24:23 [INFO] test episode 4: reward = 1.00, steps = 30
09:24:23 [INFO] test episode 5: reward = 1.00, steps = 79
09:24:23 [INFO] test episode 6: reward = 1.00, steps = 55
09:24:23 [INFO] test episode 7: reward = 1.00, steps = 92
09:24:23 [INFO] test episode 8: reward = 0.00, steps = 25
09:24:23 [INFO] test episode 9: reward = 1.00, steps = 37
09:24:23 [INFO] test episode 10: reward = 1.00, steps = 94
09:24:23 [INFO] test episode 11: reward = 1.00, steps = 14
09:24:23 [INFO] test episode 12: reward = 0.00, steps = 41
09:24:23 [INFO] test episode 13: reward = 1.00, steps = 10
09:24:23 [INFO] test episode 14: reward = 1.00, steps = 20
09:24:23 [INFO] test episode 15: reward = 0.00, steps = 63
09:24:23 [INFO] test episode 16: reward = 1.00, steps = 26
09:24:23 [INFO] test episode 17: reward = 1.00, steps = 9
09:24:23 [INFO] test episode 18: reward = 1.00, steps = 76
09:24:23 [INFO] test episode 19: reward = 0.00, steps = 100
09:24:23 [INFO] test episode 20: reward = 0.00, steps = 100
09:24:23 [INFO] test episode 21: reward = 1.00, steps = 61
09:24:23 [INFO] test episode 22: reward = 1.00, steps = 44
09:24:23 [INFO] test episode 23: reward = 1.00, steps = 11
09:24:23 [INFO] test episode 24: reward = 1.00, steps = 56
09:24:23 [INFO] test episode 25: reward = 1.00, steps = 77
09:24:23 [INFO] test episode 26: reward = 1.00, steps = 33
09:24:23 [INFO] test episode 27: reward = 1.00, steps = 60
09:24:23 [INFO] test episode 28: reward = 1.00, steps = 19
09:24:23 [INFO] test episode 29: reward = 0.00, steps = 64
09:24:23 [INFO] test episode 30: reward = 1.00, steps = 92
09:24:23 [INFO] test episode 31: reward = 0.00, steps = 100
09:24:23 [INFO] test episode 32: reward = 1.00, steps = 19
09:24:23 [INFO] test episode 33: reward = 1.00, steps = 19
09:24:23 [INFO] test episode 34: reward = 1.00, steps = 20
09:24:23 [INFO] test episode 35: reward = 0.00, steps = 100
09:24:23 [INFO] test episode 36: reward = 0.00, steps = 40
09:24:23 [INFO] test episode 37: reward = 1.00, steps = 26
09:24:23 [INFO] test episode 38: reward = 1.00, steps = 9
09:24:23 [INFO] test episode 39: reward = 1.00, steps = 53
09:24:23 [INFO] test episode 40: reward = 1.00, steps = 40
09:24:23 [INFO] test episode 41: reward = 1.00, steps = 45
09:24:23 [INFO] test episode 42: reward = 0.00, steps = 99
09:24:23 [INFO] test episode 43: reward = 1.00, steps = 76
09:24:23 [INFO] test episode 44: reward = 1.00, steps = 10
09:24:23 [INFO] test episode 45: reward = 1.00, steps = 13
09:24:23 [INFO] test episode 46: reward = 1.00, steps = 17
09:24:23 [INFO] test episode 47: reward = 1.00, steps = 72
09:24:23 [INFO] test episode 48: reward = 0.00, steps = 8
09:24:23 [INFO] test episode 49: reward = 0.00, steps = 31
09:24:23 [INFO] test episode 50: reward = 1.00, steps = 65
09:24:23 [INFO] test episode 51: reward = 1.00, steps = 41
09:24:23 [INFO] test episode 52: reward = 0.00, steps = 12
09:24:23 [INFO] test episode 53: reward = 1.00, steps = 31
09:24:23 [INFO] test episode 54: reward = 0.00, steps = 23
09:24:23 [INFO] test episode 55: reward = 1.00, steps = 41
09:24:23 [INFO] test episode 56: reward = 1.00, steps = 17
09:24:23 [INFO] test episode 57: reward = 1.00, steps = 97
09:24:23 [INFO] test episode 58: reward = 0.00, steps = 64
09:24:23 [INFO] test episode 59: reward = 1.00, steps = 32
09:24:23 [INFO] test episode 60: reward = 1.00, steps = 50
09:24:23 [INFO] test episode 61: reward = 0.00, steps = 75
09:24:23 [INFO] test episode 62: reward = 1.00, steps = 9
09:24:23 [INFO] test episode 63: reward = 1.00, steps = 20
09:24:23 [INFO] test episode 64: reward = 1.00, steps = 75
09:24:23 [INFO] test episode 65: reward = 1.00, steps = 12
09:24:23 [INFO] test episode 66: reward = 1.00, steps = 45
09:24:23 [INFO] test episode 67: reward = 1.00, steps = 35
09:24:23 [INFO] test episode 68: reward = 1.00, steps = 20
09:24:23 [INFO] test episode 69: reward = 0.00, steps = 95
09:24:23 [INFO] test episode 70: reward = 1.00, steps = 32
09:24:23 [INFO] test episode 71: reward = 0.00, steps = 37
09:24:23 [INFO] test episode 72: reward = 1.00, steps = 8
09:24:23 [INFO] test episode 73: reward = 1.00, steps = 33
09:24:23 [INFO] test episode 74: reward = 1.00, steps = 28
09:24:23 [INFO] test episode 75: reward = 0.00, steps = 36
09:24:23 [INFO] test episode 76: reward = 1.00, steps = 33
09:24:23 [INFO] test episode 77: reward = 1.00, steps = 20
09:24:23 [INFO] test episode 78: reward = 1.00, steps = 51
09:24:23 [INFO] test episode 79: reward = 0.00, steps = 19
09:24:23 [INFO] test episode 80: reward = 1.00, steps = 26
09:24:23 [INFO] test episode 81: reward = 0.00, steps = 40
09:24:23 [INFO] test episode 82: reward = 1.00, steps = 19
09:24:23 [INFO] test episode 83: reward = 1.00, steps = 9
09:24:23 [INFO] test episode 84: reward = 0.00, steps = 10
09:24:23 [INFO] test episode 85: reward = 1.00, steps = 8
09:24:23 [INFO] test episode 86: reward = 1.00, steps = 99
09:24:23 [INFO] test episode 87: reward = 0.00, steps = 100
09:24:23 [INFO] test episode 88: reward = 1.00, steps = 56
09:24:23 [INFO] test episode 89: reward = 1.00, steps = 41
09:24:23 [INFO] test episode 90: reward = 1.00, steps = 31
09:24:23 [INFO] test episode 91: reward = 1.00, steps = 95
09:24:23 [INFO] test episode 92: reward = 1.00, steps = 38
09:24:23 [INFO] test episode 93: reward = 1.00, steps = 70
09:24:23 [INFO] test episode 94: reward = 1.00, steps = 7
09:24:23 [INFO] test episode 95: reward = 1.00, steps = 58
09:24:23 [INFO] test episode 96: reward = 1.00, steps = 44
09:24:23 [INFO] test episode 97: reward = 1.00, steps = 38
09:24:23 [INFO] test episode 98: reward = 1.00, steps = 11
09:24:23 [INFO] test episode 99: reward = 1.00, steps = 15
09:24:23 [INFO] average episode reward = 0.76 ± 0.43
```