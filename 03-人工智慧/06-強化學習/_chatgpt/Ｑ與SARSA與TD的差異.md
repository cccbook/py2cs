

* [Ｑ與SARSA與TD的差異](https://chatgpt.com/c/672adadf-1214-8012-9eef-16985976a352)

Q-learning 的特點是使用下一狀態的最大 Q 值進行更新，因此屬於策略無關方法。這意味著無論當前使用的是何種策略，Q-learning 的更新只關注如何使 Q 值趨向最優。


SARSA 在更新時考慮的是根據當前策略選擇的下一動作 a' 的 Q 值，而不是最大 Q 值。因此 SARSA 會更新策略下的 Q 值，而不是理論上的最優 Q 值。

* [Q-learning, SARSA, SARSA(lambda)](https://chatgpt.com/c/672da875-6be4-8012-a637-d81121396dae)
