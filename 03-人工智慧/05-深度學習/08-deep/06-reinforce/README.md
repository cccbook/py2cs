## Reinforcement Learning

參考書 -- [Reinforcement Learning: An Introduction 中文翻譯](https://hackmd.io/@shaoeChen/Syp8clcd_/)

## 01-cartpole

學習平衡倒單擺

## 02-gomoku zero

學習五子棋 （透過自我對下）

* 來源 -- https://github.com/junxiaosong/AlphaZero_Gomoku

學習：(可不做，直接用現成模型下棋) 

```
$ pip install theano
$ pip install lasagne
$ pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip
$ python train.py
(env) mac020:02-gomoku mac020$ python train.py
batch i:1, episode_len:18
batch i:2, episode_len:12
batch i:3, episode_len:17
batch i:4, episode_len:11
batch i:5, episode_len:23
kl:0.00701,lr_multiplier:1.500,loss:4.593109962504368,entropy:3.5793810332667495,explained_var_old:-0.002,explained_var_new:0.025
batch i:6, episode_len:21
kl:0.00492,lr_multiplier:2.250,loss:4.4443461876690735,entropy:3.57293974145794,explained_var_old:0.029,explained_var_new:0.171
batch i:7, episode_len:12
kl:0.00653,lr_multiplier:3.375,loss:4.480467121157454,entropy:3.5680819698988726,explained_var_old:0.094,explained_var_new:0.155
batch i:8, episode_len:23
kl:0.02258,lr_multiplier:3.375,loss:4.24182556297472,entropy:3.5557748948827896,explained_var_old:0.180,explained_var_new:0.375
batch i:9, episode_len:14
kl:0.01952,lr_multiplier:3.375,loss:4.261341882802109,entropy:3.564255769646123,explained_var_old:0.219,explained_var_new:0.330
batch i:10, episode_len:12
kl:0.02502,lr_multiplier:3.375,loss:4.232119478541061,entropy:3.540097112181642,explained_var_old:0.180,explained_var_new:0.344
batch i:11, episode_len:15
kl:0.04303,lr_multiplier:2.250,loss:4.15392869362957,entropy:3.5069655067044945,explained_var_old:0.259,explained_var_new:0.379
batch i:12, episode_len:14
kl:0.04324,lr_multiplier:1.500,loss:4.168802974937632,entropy:3.4397352653714623,explained_var_old:0.180,explained_var_new:0.314
batch i:13, episode_len:19
...
```

直接下棋，輸入格式為： x,y

```
(env) mac020:02-gomoku mac020$ python human_play.py
Player 1 with X
Player 2 with O

       0       1       2       3       4       5       6       7

   7   _       _       _       _       _       _       _       _    


   6   _       _       _       _       _       _       _       _    


   5   _       _       _       _       _       _       _       _    


   4   _       _       _       _       _       _       _       _    


   3   _       _       _       _       _       _       _       _    


   2   _       _       _       _       _       _       _       _    


   1   _       _       _       _       _       _       _       _    


   0   _       _       _       _       _       _       _       _    


Player 1 with X
Player 2 with O

       0       1       2       3       4       5       6       7

   7   _       _       _       _       _       _       _       _    


   6   _       _       _       _       _       _       _       _    


   5   _       _       _       _       _       _       _       _    


   4   _       _       _       _       _       _       _       _    


   3   _       _       _       _       O       _       _       _    


   2   _       _       _       _       _       _       _       _    


   1   _       _       _       _       _       _       _       _    


   0   _       _       _       _       _       _       _       _    


Your move: 4,4
Player 1 with X
Player 2 with O

       0       1       2       3       4       5       6       7

   7   _       _       _       _       _       _       _       _    


   6   _       _       _       _       _       _       _       _    


   5   _       _       _       _       _       _       _       _    


   4   _       _       _       _       X       _       _       _    


   3   _       _       _       _       O       _       _       _    


   2   _       _       _       _       _       _       _       _    


   1   _       _       _       _       _       _       _       _    


   0   _       _       _       _       _       _       _       _    


Player 1 with X
Player 2 with O

       0       1       2       3       4       5       6       7

   7   _       _       _       _       _       _       _       _    


   6   _       _       _       _       _       _       _       _    


   5   _       _       _       _       _       _       _       _    


   4   _       _       _       O       X       _       _       _    


   3   _       _       _       _       O       _       _       _    


   2   _       _       _       _       _       _       _       _    


   1   _       _       _       _       _       _       _       _    


   0   _       _       _       _       _       _       _       _    


Your move: 5,2
Player 1 with X
Player 2 with O

       0       1       2       3       4       5       6       7

   7   _       _       _       _       _       _       _       _    


   6   _       _       _       _       _       _       _       _    


   5   _       _       X       _       _       _       _       _    


   4   _       _       _       O       X       _       _       _    


   3   _       _       _       _       O       _       _       _    


   2   _       _       _       _       _       _       _       _    


   1   _       _       _       _       _       _       _       _    


   0   _       _       _       _       _       _       _       _    


Player 1 with X
Player 2 with O

       0       1       2       3       4       5       6       7

   7   _       _       _       _       _       _       _       _    


   6   _       _       _       _       _       _       _       _    


   5   _       _       X       _       _       _       _       _    


   4   _       _       _       O       X       _       _       _    


   3   _       _       _       _       O       _       _       _    


   2   _       _       _       _       _       O       _       _    


   1   _       _       _       _       _       _       _       _    


   0   _       _       _       _       _       _       _       _    


Your move: 1,6
Player 1 with X
Player 2 with O

       0       1       2       3       4       5       6       7

   7   _       _       _       _       _       _       _       _    


   6   _       _       _       _       _       _       _       _    


   5   _       _       X       _       _       _       _       _    


   4   _       _       _       O       X       _       _       _    


   3   _       _       _       _       O       _       _       _    


   2   _       _       _       _       _       O       _       _    


   1   _       _       _       _       _       _       X       _    


   0   _       _       _       _       _       _       _       _    


Player 1 with X
Player 2 with O

       0       1       2       3       4       5       6       7

   7   _       _       _       _       _       _       _       _    


   6   _       _       _       _       _       _       _       _    


   5   _       _       X       _       _       _       _       _    


   4   _       _       _       O       X       _       _       _    


   3   _       _       _       O       O       _       _       _    


   2   _       _       _       _       _       O       _       _    


   1   _       _       _       _       _       _       X       _    


   0   _       _       _       _       _       _       _       _    


Your move: 3,2
Player 1 with X
Player 2 with O

       0       1       2       3       4       5       6       7

   7   _       _       _       _       _       _       _       _    


   6   _       _       _       _       _       _       _       _    


   5   _       _       X       _       _       _       _       _    


   4   _       _       _       O       X       _       _       _    


   3   _       _       X       O       O       _       _       _    


   2   _       _       _       _       _       O       _       _    


   1   _       _       _       _       _       _       X       _    


   0   _       _       _       _       _       _       _       _    


Player 1 with X
Player 2 with O

       0       1       2       3       4       5       6       7

   7   _       _       _       _       _       _       _       _    


   6   _       _       _       _       _       _       _       _    


   5   _       _       X       _       _       _       _       _    


   4   _       _       _       O       X       _       _       _    


   3   _       _       X       O       O       _       _       _    


   2   _       _       _       O       _       O       _       _    


   1   _       _       _       _       _       _       X       _    


   0   _       _       _       _       _       _       _       _    


Your move: 5,3
Player 1 with X
Player 2 with O

       0       1       2       3       4       5       6       7

   7   _       _       _       _       _       _       _       _    


   6   _       _       _       _       _       _       _       _    


   5   _       _       X       X       _       _       _       _    


   4   _       _       _       O       X       _       _       _    


   3   _       _       X       O       O       _       _       _    


   2   _       _       _       O       _       O       _       _    


   1   _       _       _       _       _       _       X       _    


   0   _       _       _       _       _       _       _       _    


Player 1 with X
Player 2 with O

       0       1       2       3       4       5       6       7

   7   _       _       _       _       _       _       _       _    


   6   _       _       _       _       _       _       _       _    


   5   _       _       X       X       _       _       _       _    


   4   _       _       _       O       X       _       _       _    


   3   _       _       X       O       O       _       _       _    


   2   _       _       _       O       _       O       _       _    


   1   _       _       _       O       _       _       X       _    


   0   _       _       _       _       _       _       _       _    


Your move: 0,3
Player 1 with X
Player 2 with O

       0       1       2       3       4       5       6       7

   7   _       _       _       _       _       _       _       _    


   6   _       _       _       _       _       _       _       _    


   5   _       _       X       X       _       _       _       _    


   4   _       _       _       O       X       _       _       _    


   3   _       _       X       O       O       _       _       _    


   2   _       _       _       O       _       O       _       _    


   1   _       _       _       O       _       _       X       _    


   0   _       _       _       X       _       _       _       _    


Player 1 with X
Player 2 with O

       0       1       2       3       4       5       6       7

   7   _       _       _       _       _       _       _       _    


   6   _       _       _       _       _       _       _       _    


   5   _       _       X       X       _       _       _       _    


   4   _       _       _       O       X       _       _       _    


   3   _       _       X       O       O       O       _       _    


   2   _       _       _       O       _       O       _       _    


   1   _       _       _       O       _       _       X       _    


   0   _       _       _       X       _       _       _       _    


Your move: 3,6
Player 1 with X
Player 2 with O

       0       1       2       3       4       5       6       7

   7   _       _       _       _       _       _       _       _    


   6   _       _       _       _       _       _       _       _    


   5   _       _       X       X       _       _       _       _    


   4   _       _       _       O       X       _       _       _    


   3   _       _       X       O       O       O       X       _    


   2   _       _       _       O       _       O       _       _    


   1   _       _       _       O       _       _       X       _    


   0   _       _       _       X       _       _       _       _    


Player 1 with X
Player 2 with O

       0       1       2       3       4       5       6       7

   7   _       _       _       _       _       _       _       _    


   6   _       _       _       _       _       _       _       _    


   5   _       _       X       X       _       _       _       _    


   4   _       _       _       O       X       O       _       _    


   3   _       _       X       O       O       O       X       _    


   2   _       _       _       O       _       O       _       _    


   1   _       _       _       O       _       _       X       _    


   0   _       _       _       X       _       _       _       _    


Your move: 5,5
Player 1 with X
Player 2 with O

       0       1       2       3       4       5       6       7

   7   _       _       _       _       _       _       _       _    


   6   _       _       _       _       _       _       _       _    


   5   _       _       X       X       _       X       _       _    


   4   _       _       _       O       X       O       _       _    


   3   _       _       X       O       O       O       X       _    


   2   _       _       _       O       _       O       _       _    


   1   _       _       _       O       _       _       X       _    


   0   _       _       _       X       _       _       _       _    


Player 1 with X
Player 2 with O

       0       1       2       3       4       5       6       7

   7   _       _       _       _       _       _       _       _    


   6   _       _       _       _       _       _       _       _    


   5   _       _       X       X       _       X       O       _    


   4   _       _       _       O       X       O       _       _    


   3   _       _       X       O       O       O       X       _    


   2   _       _       _       O       _       O       _       _    


   1   _       _       _       O       _       _       X       _    


   0   _       _       _       X       _       _       _       _    


Your move: 6,7
Player 1 with X
Player 2 with O

       0       1       2       3       4       5       6       7

   7   _       _       _       _       _       _       _       _    


   6   _       _       _       _       _       _       _       X    


   5   _       _       X       X       _       X       O       _    


   4   _       _       _       O       X       O       _       _    


   3   _       _       X       O       O       O       X       _    


   2   _       _       _       O       _       O       _       _    


   1   _       _       _       O       _       _       X       _    


   0   _       _       _       X       _       _       _       _    


Player 1 with X
Player 2 with O

       0       1       2       3       4       5       6       7

   7   _       _       _       _       _       _       _       _    


   6   _       _       _       _       _       _       _       X    


   5   _       _       X       X       _       X       O       _    


   4   _       _       _       O       X       O       _       _    


   3   _       _       X       O       O       O       X       _    


   2   _       _       _       O       _       O       _       _    


   1   _       _       _       O       _       O       X       _    


   0   _       _       _       X       _       _       _       _    


Your move: 1,2
Player 1 with X
Player 2 with O

       0       1       2       3       4       5       6       7

   7   _       _       _       _       _       _       _       _    


   6   _       _       _       _       _       _       _       X    


   5   _       _       X       X       _       X       O       _    


   4   _       _       _       O       X       O       _       _    


   3   _       _       X       O       O       O       X       _    


   2   _       _       _       O       _       O       _       _    


   1   _       _       X       O       _       O       X       _    


   0   _       _       _       X       _       _       _       _    


Player 1 with X
Player 2 with O

       0       1       2       3       4       5       6       7

   7   _       _       _       _       _       _       _       _    


   6   _       _       _       _       _       _       _       X    


   5   _       _       X       X       _       X       O       _    


   4   _       _       _       O       X       O       _       _    


   3   _       _       X       O       O       O       X       _    


   2   _       _       _       O       _       O       _       _    


   1   _       _       X       O       _       O       X       _    


   0   _       _       _       X       _       O       _       _    


Game end. Winner is MCTS 2

```