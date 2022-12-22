# 環境 gym 當中的一些問題描述

## CartPole 車子上的竿子平衡問題

* https://www.gymlibrary.dev/environments/classic_control/cart_pole/

![](https://www.gymlibrary.dev/_images/cart_pole.gif)

## Mountain Car 小車如何脫離谷底的問題

* https://www.gymlibrary.dev/environments/classic_control/mountain_car/

![](https://www.gymlibrary.dev/_images/mountain_car.gif)

## FrozenLake 冰凍的湖如何穿越問題

* https://www.gymlibrary.dev/environments/toy_text/frozen_lake/

![](https://www.gymlibrary.dev/_images/frozen_lake.gif)

* https://blog.csdn.net/hba646333407/article/details/104433054

FrozenLake 是一个 4*4 的网络格子，每个格子可以是起始块，目标块、冻结块或者危险块。我们的目标是让 agent 学习从开始块如何行动到目标块上，而不是移动到危险块上。agent 可以选择向上、向下、向左或者向右移动，同时游戏中还有可能吹来一阵风，将 agent 吹到任意的方块上。在这种情况下，每个时刻都有完美的策略是不能的，但是如何避免危险洞并且到达目标洞肯定是可行的。

更通俗一点地讲，就是冬天来了，你和你的朋友在公园里玩飞盘的时候，你把飞盘扔到了湖中央。水大部分都已冻结，但也有一些地方融化出了几个洞。如果你踏进其中一个洞，你就会掉进冰冷的水里。在这个时候，由于没有其他的飞盘，所以必须穿过湖面并取回光盘。然而，冰是滑的，所以你不会总是朝着你想要的方向前进。

## Bipedal Walker: 双足行走机器人

![](https://www.gymlibrary.dev/_images/bipedal_walker.gif)


v3: returns closest lidar trace instead of furthest; faster video recording

v2: Count energy spent

v1: Legs now report contact with ground; motors have higher torque and speed; ground has higher friction; lidar rendered less nervously.

v0: Initial version

## Car Racing : 賽車

* https://www.gymlibrary.dev/environments/box2d/car_racing/

![](https://www.gymlibrary.dev/_images/car_racing.gif)
