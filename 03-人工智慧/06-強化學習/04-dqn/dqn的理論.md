


![](./img/dqn1.png)



![](https://datawhalechina.github.io/easy-rl/img/ch6/6.19.png)


* [6.6 深度Q网络](https://datawhalechina.github.io/easy-rl/#/chapter6/chapter6?id=_66-%e6%b7%b1%e5%ba%a6q%e7%bd%91%e7%bb%9c)

Q：深度Q网络 和 Q学习 有什么不同？

A：整体来说，深度Q网络与Q学习的目标价值以及价值的更新方式都非常相似。主要的不同点在于：深度Q网络 将Q学习与深度学习结合，用深度网络来近似动作价值函数，而 Q学习 则是采用表格存储；深度Q网络采用了经验回放的训练方法，从历史数据中随机采样，而Q学习直接采用下一个状态的数据进行学习。
