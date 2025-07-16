import pytest
import torch
import dtorch

def test_maxpool2d():
    # 測試數據
    x = torch.randn(1, 1, 4, 4)  # 輸入張量，形狀為 (batch_size, channels, height, width)

    # 使用手動實現的 MaxPool2d
    maxpool_custom = dtorch.nn.MaxPool2d(kernel_size=2, stride=2)
    output_custom = maxpool_custom(x)

    # 使用 PyTorch 內建的 MaxPool2d
    maxpool_builtin = torch.nn.MaxPool2d(kernel_size=2, stride=2)
    output_builtin = maxpool_builtin(x)

    # 比較結果
    print("手動實現的 MaxPool2d 輸出:", output_custom)
    print("PyTorch 內建的 MaxPool2d 輸出:", output_builtin)
    print("結果是否一致:", torch.allclose(output_custom, output_builtin))
    assert torch.allclose(output_custom, output_builtin), "輸出不一致"

def test_conv2d():
    # 測試數據
    x = torch.randn(1, 1, 4, 4)  # 輸入張量，形狀為 (batch_size, in_channels, height, width)

    # 使用手動實現的 Conv2d
    conv_custom = dtorch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True)
    output_custom = conv_custom(x)

    # 使用 PyTorch 內建的 Conv2d
    conv_builtin = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True)
    # 將手動實現的權重和偏置複製到內建的 Conv2d
    conv_builtin.weight.data = conv_custom.weight.data
    conv_builtin.bias.data = conv_custom.bias.data
    output_builtin = conv_builtin(x)

    # 比較結果
    print("手動實現的 Conv2d 輸出:", output_custom)
    print("PyTorch 內建的 Conv2d 輸出:", output_builtin)
    print("結果是否一致:", torch.allclose(output_custom, output_builtin, atol=1e-5))
    assert torch.allclose(output_custom, output_builtin), "輸出不一致"
