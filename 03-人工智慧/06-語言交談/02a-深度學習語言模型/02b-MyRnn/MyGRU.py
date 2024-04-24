import torch

class MyGRU:
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=False):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first

        # 初始化GRU層的權重和偏置
        self.weight_ih = torch.randn(3 * hidden_size, input_size)
        self.weight_hh = torch.randn(3 * hidden_size, hidden_size)
        if bias:
            self.bias_ih = torch.randn(3 * hidden_size)
            self.bias_hh = torch.randn(3 * hidden_size)
        else:
            self.bias_ih = None
            self.bias_hh = None

    def forward(self, input, h_0=None):
        if self.batch_first:
            input = input.transpose(0, 1)

        batch_size, seq_length, _ = input.size()
        if h_0 is None:
            h_t = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        else:
            h_t = h_0

        outputs = []

        for t in range(seq_length):
            x_t = input[:, t, :]

            for layer in range(self.num_layers):
                # 隱藏層輸入線性組合
                if self.bias:
                    ih = torch.matmul(x_t, self.weight_ih.T) + self.bias_ih
                    hh = torch.matmul(h_t[layer], self.weight_hh.T) + self.bias_hh
                else:
                    ih = torch.matmul(x_t, self.weight_ih.T)
                    hh = torch.matmul(h_t[layer], self.weight_hh.T)

                # 將線性組合結果分成三部分：重置門、更新門、新的候選隱藏狀態
                r_t, z_t, n_t = torch.split(ih + hh, self.hidden_size, dim=1)

                # 選擇Sigmoid作為重置門和更新門的激活函數
                r_t = torch.sigmoid(r_t)
                z_t = torch.sigmoid(z_t)

                # 使用tanh作為新的候選隱藏狀態的激活函數
                n_t = torch.tanh(n_t)

                # 更新隱藏狀態
                h_t[layer] = (1 - z_t) * n_t + z_t * h_t[layer] * r_t

                x_t = h_t[layer]

            outputs.append(h_t[-1].clone())

        outputs = torch.stack(outputs, dim=1)

        return outputs, h_t

# 測試自行實作的GRU層
if __name__ == "__main__":
    # 建立一個輸入張量（假設是一個batch大小為2、序列長度為3、輸入維度為4的序列）
    input = torch.tensor([[[1.0, 2.0, 3.0, 4.0],
                           [5.0, 6.0, 7.0, 8.0],
                           [9.0, 10.0, 11.0, 12.0]],
                          [[13.0, 14.0, 15.0, 16.0],
                           [17.0, 18.0, 19.0, 20.0],
                           [21.0, 22.0, 23.0, 24.0]]])

    # 建立自行實作的GRU層（輸入維度為4，隱藏層大小為3，層數為1）
    gru_layer = MyGRU(input_size=4, hidden_size=3)

    # 將輸入張量通過自行實作的GRU層
    output, h_n = gru_layer.forward(input)

    print("輸入張量:")
    print(input)
    print("自行實作的GRU層的輸出:")
    print(output)
    print("最後一個隱藏狀態:")
    print(h_n)
