import pytest
import torch
import dtorch

def test_cross_entropy_loss():
    # 測試數據
    logits = torch.tensor([[2.0, 1.0, 0.1], [0.5, 2.0, 0.3]], requires_grad=True)  # 模型輸出
    target = torch.tensor([0, 1])  # 目標類別

    # 使用自定義的 CrossEntropyLoss
    criterion_custom = dtorch.nn.CrossEntropyLoss()
    loss_custom = criterion_custom(logits, target)

    # 使用 PyTorch 內建的 CrossEntropyLoss
    criterion_builtin = torch.nn.CrossEntropyLoss()
    loss_builtin = criterion_builtin(logits, target)

    # 比較結果
    print("自定義 CrossEntropyLoss 結果:", loss_custom.item())
    print("PyTorch 內建 CrossEntropyLoss 結果:", loss_builtin.item())
    print("結果是否一致:", torch.allclose(loss_custom, loss_builtin))
    # assert torch.allclose(loss_custom, logits), "輸出不一致"
    assert torch.allclose(loss_custom, loss_builtin), "輸出不一致"
