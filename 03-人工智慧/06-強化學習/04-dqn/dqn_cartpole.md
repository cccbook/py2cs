# dqn_cartpole.py

來源 -- https://github.com/datawhalechina/easy-rl/blob/master/notebooks/DQN.ipynb

實驗結果：從一開始的獎勵 10 到後來大約 170，是有逐步提升 ..

## 執行結果

```
$ python dqn_cartpole.py
超参数
================================================================================
        Name                   Value                    Type
     algo_name                  DQN                <class 'str'>
      env_name              CartPole-v1            <class 'str'>
     train_eps                  200                <class 'int'>
      test_eps                   20                <class 'int'>
    ep_max_steps               100000              <class 'int'>
       gamma                    0.95              <class 'float'>
   epsilon_start                0.95              <class 'float'>
    epsilon_end                 0.01              <class 'float'>
   epsilon_decay                500                <class 'int'>
         lr                    0.0001             <class 'float'>
  memory_capacity              100000              <class 'int'>
     batch_size                  64                <class 'int'>
   target_update                 4                 <class 'int'>
     hidden_dim                 256                <class 'int'>
       device                   cpu                <class 'str'>
        seed                     10                <class 'int'>
================================================================================
状态空间维度：4，动作空间维度：2
开始训练！
回合：10/200，奖励：10.00，Epislon：0.603
回合：20/200，奖励：11.00，Epislon：0.479
回合：30/200，奖励：12.00，Epislon：0.370
回合：40/200，奖励：19.00，Epislon：0.295
回合：50/200，奖励：22.00，Epislon：0.236
回合：60/200，奖励：45.00，Epislon：0.090
回合：70/200，奖励：75.00，Epislon：0.038
回合：80/200，奖励：96.00，Epislon：0.017
回合：90/200，奖励：429.00，Epislon：0.010
回合：100/200，奖励：200.00，Epislon：0.010
回合：110/200，奖励：212.00，Epislon：0.010
回合：120/200，奖励：231.00，Epislon：0.010
回合：130/200，奖励：170.00，Epislon：0.010
回合：140/200，奖励：168.00，Epislon：0.010
回合：150/200，奖励：175.00，Epislon：0.010
回合：160/200，奖励：184.00，Epislon：0.010
回合：170/200，奖励：184.00，Epislon：0.010
回合：180/200，奖励：162.00，Epislon：0.010
回合：190/200，奖励：168.00，Epislon：0.010
回合：200/200，奖励：173.00，Epislon：0.010
完成训练！
开始测试！
回合：1/20，奖励：171.00
回合：2/20，奖励：176.00
回合：3/20，奖励：179.00
回合：4/20，奖励：173.00
回合：5/20，奖励：168.00
回合：6/20，奖励：179.00
回合：7/20，奖励：181.00
回合：8/20，奖励：175.00
回合：9/20，奖励：173.00
回合：10/20，奖励：211.00
回合：11/20，奖励：174.00
回合：12/20，奖励：175.00
回合：13/20，奖励：157.00
回合：14/20，奖励：159.00
回合：15/20，奖励：178.00
回合：16/20，奖励：168.00
回合：17/20，奖励：172.00
回合：18/20，奖励：176.00
回合：19/20，奖励：183.00
回合：20/20，奖励：166.00
完成测试
```

train:

![](./img/dqn_cartpole_train.png)

test:

![](./img/dqn_cartpole_test.png)


## 問題： 

ep_max_steps= 100000，因此最高應該 reward 會到 1000000

但我們的實驗結果最多到 429 ，而且是在中間 90/200 時飆高，之後反而下降了 ...

更奇怪的是，來源中總是到 200 就停了，為何呢？


