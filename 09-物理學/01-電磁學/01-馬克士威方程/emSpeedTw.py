import numpy as np
import matplotlib.pyplot as plt
import tqdm
from matplotlib.animation import FuncAnimation

# 用法：
# fdtd = fdtd1d()  # 創建模擬器
# fdtd.run()  # 執行模擬
# fdtd.plot()  # 繪製電場和磁場圖

class fdtd1d(object):
    def __init__(self, Nx = 201, dx = 1e-3, c = 1, source = 100):
        # 網格屬性：
        # 1. 網格單元數量
        self.Nx = Nx
        # 2. 網格單元大小
        self.dx = dx
        # 3. 光速
        self.c = c
        # 4. 時間步長（遵循Courant數）
        self.dt = dx / (1.0 * self.c)

        # 源屬性：
        # 1. 源位置的網格索引
        self.source = source
        # 2. 源的頻率、周期等：
        frequency = 1 / (self.dt * 10)
        T0 = 1.0 / frequency
        tc = 5 * T0 / 2
        self.sig = tc / 2 / np.sqrt(2 * np.log(2))

        # 初始化網格：
        # 1. 電場網格
        self.E_y = np.zeros(Nx)
        # 2. 磁場網格
        self.H_z = np.zeros(Nx - 1)

        # Mur 吸收邊界條件 (ABC)
        # 右側的ABC
        self.E_y_h = 0 
        # 左側的ABC
        self.E_y_l = 0

        # 記錄隨時間變化的電場
        self.E_t = []
        
        # 用於繪圖的物理網格
        self.x = np.arange(0, Nx, 1)
        self.Dx = np.arange(0.5, Nx-0.5, 1)

    def run(self, n_iter = 180):
        # 主FDTD迴圈
        dt = self.dt
        dx = self.dx
        c = self.c
        sig = self.sig
        source = self.source
        
        for n in tqdm.trange(n_iter):
            # 更新磁場
            self.H_z = self.H_z - dt / dx * (self.E_y[1:] - self.E_y[:-1])     
    
            # 更新電場
            self.E_y[1:-1] = self.E_y[1:-1] - dt / dx * (self.H_z[1:] - self.H_z[:-1])
        
            # 啟動源以開始模擬
            pulse = np.exp((-((n+1) * dt - 3 * np.sqrt(2) * sig)**2) / (2 * sig**2))
            self.E_y[source] = self.E_y[source] + pulse
         
            # 右側的吸收邊界條件
            self.E_y[-1] = self.E_y_h + (c * dt - dx) / (c * dt + dx) * (self.E_y[-2] - self.E_y[-1])
            self.E_y_h = self.E_y[-2]
    
            # 左側的吸收邊界條件
            self.E_y[0] = self.E_y_l + (c * dt - dx) / (c * dt + dx) * (self.E_y[1] - self.E_y[0])
            self.E_y_l = self.E_y[1]
            
            self.E_t.append(self.E_y.copy())
        
    def plot(self):
        # 繪製 E_y 和 H_z 的空間分佈
        plt.figure(figsize = (10, 5))
        plt.subplot(2, 1, 1)
        plt.plot(self.x, self.E_y)
        plt.ylabel("E_y")
        plt.grid(True)
        plt.subplot(2, 1, 2)
        plt.plot(self.Dx, self.H_z)
        plt.ylabel("H_x")
        plt.grid('on')
        plt.xlabel("x")
        plt.show()
        
    def animate(self, file_dir = "fdtd_1d_animation.gif", N = 500):
        # 動畫化 self.Et 為 .gif 檔案
        # N: 要保存為 .gif 動畫的總步數
        Et = self.E_t[-N:]
        
        fig, ax = plt.subplots()
        ax.set(xlim = [-10, 210], ylim = [-1, 1])
        line = ax.plot(range(len(Et[0])), Et[0], color = "r", linewidth = 2)[0]
        ax.set_xlabel("x")
        ax.set_ylabel("Electric field")
        ax.grid(True)

        def animate(i):
            line.set_ydata(Et[i])

        anim = FuncAnimation(fig, animate, interval = 50, frames = len(Et) - 1)
        anim.save(file_dir, writer = "pillow") 
        plt.show()

fdtd = fdtd1d()  # 創建模擬器
fdtd.run(80)  # 執行模擬 80 步
fdtd.plot()  # 繪製電場和磁場
