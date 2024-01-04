import matplotlib.pyplot as plt

d = [1,2,3,4,5,6,7,8,9]
f = [0,1,0,0,1,0,1,1,0]

fig, axs = plt.subplots(2, 2)
axs[0,0].plot(d)
axs[0,1].plot(f)
axs[1,0].plot(d)
axs[1,1].plot(f)
plt.show()
