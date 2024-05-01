import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(1, 1000, num=1000, endpoint = True)

q = np.load('q-episode-reward.npy')
s = np.load('s-episode-reward.npy')

plt.subplot(1,2,1)
plt.plot(x, s, 'blue')
plt.title('SARSA')
plt.xlabel('Episodes')
plt.ylabel('Reward per Episode')
plt.ylim(-500, 0)

plt.subplot(1,2,2)
plt.plot(x, q, 'orange')
plt.title('Q-Learning')
plt.xlabel('Episodes')
plt.ylim(-500, 0)

plt.show()
