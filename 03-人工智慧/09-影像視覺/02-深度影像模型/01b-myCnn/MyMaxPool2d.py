import torch

class MyMaxPool2d:
    def __init__(self, kernel_size, stride=None, padding=(0,0)):
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding

        # 計算輸出特徵圖的大小
        # print('height=', height, "ph=", ph, "kh=", kh, 'sh=', sh)
        oh = (height + 2 * ph - kh) // sh + 1
        ow = (width + 2 * pw - kw) // sw + 1

        # 初始化輸出特徵圖
        output = torch.zeros(batch_size, channels, oh, ow)

        # 進行最大池化操作
        for i in range(oh):
            for j in range(ow):
                # 計算每個窗口的位置
                h_start = i * sh
                h_end = h_start + kh
                w_start = j * sw
                w_end = w_start + kw

                # 從輸入張量中提取出窗口區域
                window = x[:, :, h_start:h_end, w_start:w_end]

                # 對窗口區域進行最大池化操作，並將結果放入輸出特徵圖中
                output[:, :, i, j] = torch.max(torch.max(window, dim=2)[0], dim=2)[0]

        return output

# 測試自行實作的Max Pooling 2D層
if __name__ == "__main__":
    # 建立一個輸入張量（假設是一個具有1個通道、高度為4、寬度為4的圖像）
    x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0],
                        [5.0, 6.0, 7.0, 8.0],
                        [9.0, 10.0, 11.0, 12.0],
                        [13.0, 14.0, 15.0, 16.0]]]])

    # 建立自行實作的Max Pooling 2D層（使用2x2的kernel size）
    pool_layer = MyMaxPool2d(kernel_size=(2, 2))

    # 將輸入張量通過自行實作的Max Pooling 2D層
    output = pool_layer.forward(x)

    print("輸入張量:")
    print(x)
    print("自行實作的Max Pooling 2D後的輸出:")
    print(output)
