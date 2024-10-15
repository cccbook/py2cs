
ChatGPT: https://chatgpt.com/c/6704bf2b-597c-8012-b3ef-454b9c0d938a

ccc:

    請寫一個 python 示範傅立葉轉換

    轉換前用一個 cos(n x) 的波產生資料

    但轉換後取回 n 值到底是多少

    並且用 matplotlib 繪圖顯示之

ChatGPT:

你可以使用 `numpy` 和 `matplotlib` 來生成一個 cos 函數的波形，然後使用 `numpy.fft` 來對波形進行傅立葉轉換，最後根據傅立葉轉換結果取回對應的頻率（n 值），並用圖形顯示。

以下是具體的 Python 示範程式碼：

```python
import numpy as np
import matplotlib.pyplot as plt

# 參數設置
n = 5  # cos 波的頻率
sampling_rate = 1000  # 取樣頻率
T = 1.0  # 時間長度
t = np.linspace(0, T, int(sampling_rate * T), endpoint=False)  # 時間座標

# 生成 cos(n * x) 波形
signal = np.cos(2 * np.pi * n * t)

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
```

### 程式說明：
1. **生成 cos 波形**：使用 `np.cos(2 * np.pi * n * t)` 來生成頻率為 `n` 的 cos 波形。
2. **傅立葉轉換**：使用 `np.fft.fft` 進行快速傅立葉轉換，並使用 `np.fft.fftfreq` 來獲得對應的頻率。
3. **擷取主要頻率**：計算傅立葉轉換後的頻譜，找到頻譜中的最大值，並對應出該頻率（即 n 值）。
4. **圖形顯示**：分別顯示原始 cos 波形和傅立葉轉換後的頻譜圖，並標註出檢測到的主要頻率。
