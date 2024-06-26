import torch

class MyConv2d:
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,1), padding=(0,0)):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # 初始化卷積核權重和偏置
        self.weight = torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1])
        self.bias = torch.zeros(out_channels)

    def forward(self, x):
        batch_size, in_channels, height, width = x.size()
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding

        # 計算輸出特徵圖的大小
        oh = (height + 2 * ph - kh) // sh + 1
        ow = (width + 2 * pw - kw) // sw + 1

        # 初始化輸出特徵圖
        output = torch.zeros(batch_size, self.out_channels, oh, ow)

        # 進行卷積操作
        for i in range(oh):
            for j in range(ow):
                # 計算每個窗口的位置
                h_start = i * sh
                h_end = h_start + kh
                w_start = j * sw
                w_end = w_start + kw

                # 從輸入張量中提取出窗口區域
                window = x[:, :, h_start:h_end, w_start:w_end]

                # 對窗口區域進行卷積操作，並將結果放入輸出特徵圖中
                output[:, :, i, j] = torch.sum(window * self.weight.view(self.out_channels, -1).unsqueeze(-1).unsqueeze(-1), dim=(1, 2, 3)) + self.bias

        return output

# 測試自行實作的Conv2d層
if __name__ == "__main__":
    # 建立一個輸入張量（假設是一個具有1個通道、高度為4、寬度為4的圖像）
    x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0],
                        [5.0, 6.0, 7.0, 8.0],
                        [9.0, 10.0, 11.0, 12.0],
                        [13.0, 14.0, 15.0, 16.0]]]])

    # 建立自行實作的Conv2d層（使用1個3x3的卷積核）
    conv_layer = MyConv2d(in_channels=1, out_channels=1, kernel_size=(3, 3))

    # 將輸入張量通過自行實作的Conv2d層
    output = conv_layer.forward(x)

    print("輸入張量:")
    print(x)
    print("自行實作的Conv2d後的輸出:")
    print(output)
