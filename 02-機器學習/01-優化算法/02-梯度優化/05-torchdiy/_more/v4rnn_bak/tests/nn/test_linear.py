import pytest
import torch
import dtorch

# 定義一個 fixture 來共享輸入數據
@pytest.fixture
def input_tensor():
    return torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)

# 定義一個 fixture 來初始化 dtorch.nn.Linear
@pytest.fixture
def dtorch_linear():
    return dtorch.nn.Linear(in_features=3, out_features=2)

# 定義一個 fixture 來初始化 PyTorch 的 nn.Linear
@pytest.fixture
def builtin_linear(dtorch_linear):
    linear = torch.nn.Linear(in_features=3, out_features=2)
    # 將權重和偏置設置為與 dtorch Linear 相同
    with torch.no_grad():
        linear.weight.copy_(dtorch_linear.weight)
        linear.bias.copy_(dtorch_linear.bias)
    return linear

# 測試 dtorch.nn.Linear 的輸出是否與 PyTorch 的 nn.Linear 一致
def test_linear_output(input_tensor, dtorch_linear, builtin_linear):
    dtorch_output = dtorch_linear(input_tensor)
    builtin_output = builtin_linear(input_tensor)
    assert torch.allclose(dtorch_output, builtin_output), "輸出不一致"

# 測試 dtorch.nn.Linear 的權重梯度是否與 PyTorch 的 nn.Linear 一致
def test_linear_weight_grad(input_tensor, dtorch_linear, builtin_linear):
    # 計算 dtorch Linear 的梯度
    dtorch_output = dtorch_linear(input_tensor)
    dtorch_output.sum().backward()
    dtorch_weight_grad = dtorch_linear.weight.grad.clone()

    # 計算 PyTorch Linear 的梯度
    builtin_output = builtin_linear(input_tensor)
    builtin_output.sum().backward()
    builtin_weight_grad = builtin_linear.weight.grad.clone()

    assert torch.allclose(dtorch_weight_grad, builtin_weight_grad), "權重梯度不一致"

# 測試 dtorch.nn.Linear 的偏置梯度是否與 PyTorch 的 nn.Linear 一致
def test_linear_bias_grad(input_tensor, dtorch_linear, builtin_linear):
    # 計算 dtorch Linear 的梯度
    dtorch_output = dtorch_linear(input_tensor)
    dtorch_output.sum().backward()
    dtorch_bias_grad = dtorch_linear.bias.grad.clone()

    # 計算 PyTorch Linear 的梯度
    builtin_output = builtin_linear(input_tensor)
    builtin_output.sum().backward()
    builtin_bias_grad = builtin_linear.bias.grad.clone()

    assert torch.allclose(dtorch_bias_grad, builtin_bias_grad), "偏置梯度不一致"
