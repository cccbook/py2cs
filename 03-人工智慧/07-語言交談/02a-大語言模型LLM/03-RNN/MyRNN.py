import torch

class MyRNN:
    def __init__(self, input_size, hidden_size, num_layers=1, nonlinearity='tanh', bias=True, batch_first=False):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity
        self.bias = bias
        self.batch_first = batch_first

        # 初始化RNN層的權重和偏置
        self.weight_ih = torch.randn(num_layers, hidden_size, input_size)
        self.weight_hh = torch.randn(num_layers, hidden_size, hidden_size)
        if bias:
            self.bias_ih = torch.randn(num_layers, hidden_size)
            self.bias_hh = torch.randn(num_layers, hidden_size)
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
                    ih = torch.matmul(x_t, self.weight_ih[layer].T) + self.bias_ih[layer]
                    hh = torch.matmul(h_t[layer], self.weight_hh[layer].T) + self.bias_hh[layer]
                else:
                    ih = torch.matmul(x_t, self.weight_ih[layer].T)
                    hh = torch.matmul(h_t[layer], self.weight_hh[layer].T)

                # 選擇非線性激活函數
                if self.nonlinearity == 'tanh':
                    h_t[layer] = torch.tanh(ih + hh)
                elif self.nonlinearity == 'relu':
                    h_t[layer] = torch.relu(ih + hh)
                else:
                    raise ValueError("Unsupported nonlinearity: {}".format(self.nonlinearity))

                x_t = h_t[layer]

            outputs.append(h_t[-1].clone())

        outputs = torch.stack(outputs, dim=1)

        return outputs, h_t

# 測試自行實作的RNN層
if __name__ == "__main__":
    # 建立一個輸入張量（假設是一個batch大小為2、序列長度為3、輸入維度為4的序列）
    input = torch.tensor([[[1.0, 2.0, 3.0, 4.0],
                           [5.0, 6.0, 7.0, 8.0],
                           [9.0, 10.0, 11.0, 12.0]],
                          [[13.0, 14.0, 15.0, 16.0],
                           [17.0, 18.0, 19.0, 20.0],
                           [21.0, 22.0, 23.0, 24.0]]])

    # 建立自行實作的RNN層（輸入維度為4，隱藏層大小為3，層數為1）
    rnn_layer = MyRNN(input_size=4, hidden_size=3)

    # 將輸入張量通過自行實作的RNN層
    output, h_n = rnn_layer.forward(input)

    print("輸入張量:")
    print(input)
    print("自行實作的RNN層的輸出:")
    print(output)
    print("最後一個隱藏狀態:")
