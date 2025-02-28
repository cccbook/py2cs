

```sh
(py310) cccimac@cccimacdeiMac 02-language % ./test_gru_fail.sh
zsh: permission denied: ./test_gru_fail.sh
(py310) cccimac@cccimacdeiMac 02-language % chmod +x test_gru_fail.sh
(py310) cccimac@cccimacdeiMac 02-language % ./test_gru_fail.sh       
tokens= 5231
len(ids)= 5231
ids.size(0)= 5231
batch_size= 20
num_batches= 261
len(ids)= 5220
ids.shape= torch.Size([20, 261])
dictionary= {0: 'the', 1: 'little', 2: 'pig', 3: '<eos>', 4: 'every', 5: 'white', 6: 'cat', 7: 'chase', 8: 'a', 9: 'bite', 10: 'black', 11: 'dog', 12: 'love'}
vocab_size= 13
training ...
Traceback (most recent call last):
  File "/Users/cccimac/Desktop/ccc/py2cs/02-機器學習/01-優化算法/02-梯度優化/05-dtorch/v5transformer/examples/02-language/main.py", line 169, in <module>
    train(corpus, method)
  File "/Users/cccimac/Desktop/ccc/py2cs/02-機器學習/01-優化算法/02-梯度優化/05-dtorch/v5transformer/examples/02-language/main.py", line 89, in train
    outputs, states = model(inputs, states) # 用 model 計算預測詞
  File "/opt/homebrew/Caskroom/miniforge/base/envs/py310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/opt/homebrew/Caskroom/miniforge/base/envs/py310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/Users/cccimac/Desktop/ccc/py2cs/02-機器學習/01-優化算法/02-梯度優化/05-dtorch/v5transformer/examples/02-language/main.py", line 57, in forward
    out, h = self.rnn(x, h)
  File "/opt/homebrew/Caskroom/miniforge/base/envs/py310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/opt/homebrew/Caskroom/miniforge/base/envs/py310/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/opt/homebrew/Caskroom/miniforge/base/envs/py310/lib/python3.10/site-packages/dtorch/gru.py", line 231, in forward
    h_n[idx] = h_t
RuntimeError: expand(torch.FloatTensor{[20, 32]}, size=[32]): the number of sizes provided (1) must be greater or equal to the number of dimensions in the tensor (2)
tokens= 5231
len(ids)= 5231
ids.size(0)= 5231
batch_size= 20
num_batches= 261
len(ids)= 5220
ids.shape= torch.Size([20, 261])
dictionary= {0: 'the', 1: 'little', 2: 'pig', 3: '<eos>', 4: 'every', 5: 'white', 6: 'cat', 7: 'chase', 8: 'a', 9: 'bite', 10: 'black', 11: 'dog', 12: 'love'}
vocab_size= 13
Sampled [100/1000] words and save to english_gru.txt
Sampled [200/1000] words and save to english_gru.txt
Sampled [300/1000] words and save to english_gru.txt
Sampled [400/1000] words and save to english_gru.txt
Sampled [500/1000] words and save to english_gru.txt
Sampled [600/1000] words and save to english_gru.txt
Sampled [700/1000] words and save to english_gru.txt
Sampled [800/1000] words and save to english_gru.txt
Sampled [900/1000] words and save to english_gru.txt
Sampled [1000/1000] words and save to english_gru.txt
```
