```sh
(base) teacher@teacherdeiMac pytorch7seg % python pytorch7seg.py
Epoch [500/5000], Loss: 0.025084
Epoch [1000/5000], Loss: 0.025019
Epoch [1500/5000], Loss: 0.025008
Epoch [2000/5000], Loss: 0.025004
Epoch [2500/5000], Loss: 0.025003
Epoch [3000/5000], Loss: 0.025002
Epoch [3500/5000], Loss: 0.025001
Epoch [4000/5000], Loss: 0.025001
Epoch [4500/5000], Loss: 0.025001
Epoch [5000/5000], Loss: 0.025000

=== 測試結果 ===
/Users/teacher/Desktop/ccc/py2cs/02-機器學習/01-優化算法/02-梯度優化/03-梯度下降法/pytorch7seg/pytorch7seg.py:71: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
  output = model(torch.tensor(segment_input, dtype=torch.float32))
Input: [1. 1. 1. 1. 1. 1. 0.] -> Predicted: 0000
Input: [0. 1. 1. 0. 0. 0. 0.] -> Predicted: 0001
Input: [1. 1. 0. 1. 1. 0. 1.] -> Predicted: 0010
Input: [1. 1. 1. 1. 0. 0. 1.] -> Predicted: 0011
Input: [0. 1. 1. 0. 0. 1. 1.] -> Predicted: 0100
Input: [1. 0. 1. 1. 0. 1. 1.] -> Predicted: 0101
Input: [1. 0. 1. 1. 1. 1. 1.] -> Predicted: 0110
Input: [1. 1. 1. 0. 0. 0. 0.] -> Predicted: 0111
Input: [1. 1. 1. 1. 1. 1. 1.] -> Predicted: 0000
Input: [1. 1. 1. 1. 0. 1. 1.] -> Predicted: 1001
```