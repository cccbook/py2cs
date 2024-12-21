import numpy as np
import matplotlib.pyplot as plt

# 參數設置
sampling_rate = 1000  # 取樣頻率
T = 1.0  # 時間長度
t = np.linspace(0, T, int(sampling_rate * T), endpoint=False)  # 時間座標

n = 50  # cos 波的頻率
# 生成 cos(n * x) 波形
#signal = np.cos(2 * np.pi * n * t) # 只有頻率 n 的 cos 波
signal = np.cos(2 * np.pi * n * t) + 2*np.cos(2 * np.pi * 2 * n * t) 
# 頻率 n 的 cos 波疊加頻率 2n 的 cos 波

# 對信號進行傅立葉轉換
fft_result = np.fft.fft(signal)
frequencies = np.fft.fftfreq(len(signal), 1 / sampling_rate)

# 只取正頻率部分
positive_frequencies = frequencies[:len(frequencies)//2]
positive_fft_result = np.abs(fft_result[:len(fft_result)//2])

# 找到最大幅值對應的頻率 n
dominant_frequency = positive_frequencies[np.argmax(positive_fft_result)]

# 繪圖
plt.figure(figsize=(12, 6))

# 原始信號繪圖
plt.subplot(1, 2, 1)
plt.plot(t, signal)
plt.title("Original Signal: cos(2π * %d * t)" % n)
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")

# 傅立葉轉換結果繪圖
plt.subplot(1, 2, 2)
plt.plot(positive_frequencies, positive_fft_result)
plt.title("Fourier Transform")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")

# 顯示主要頻率
plt.suptitle(f"Detected Frequency: {dominant_frequency} Hz", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

print(f"The detected dominant frequency is: {dominant_frequency} Hz")
