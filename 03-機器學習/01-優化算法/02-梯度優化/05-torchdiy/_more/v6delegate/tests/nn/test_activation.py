import pytest
import torch
import dtorch

def test_relu():
    # 測試數據
    x = torch.randn(1, 1, 4, 4)  # 輸入張量，形狀為 (batch_size, channels, height, width)

    # 使用手動實現的 MaxPool2d
    custom = dtorch.nn.ReLU()
    output_custom = custom(x)

    # 使用 PyTorch 內建的 MaxPool2d
    builtin = torch.nn.ReLU()
    output_builtin = builtin(x)

    # 比較結果
    assert torch.allclose(output_custom, output_builtin), "輸出不一致"
