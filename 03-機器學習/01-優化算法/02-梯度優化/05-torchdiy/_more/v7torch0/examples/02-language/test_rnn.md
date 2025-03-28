
```sh
(py310) cccimac@cccimacdeiMac 02-language % ./test_rnn.sh            
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
Epoch [1/20], Step[0/8], Loss: 2.5892, Perplexity: 13.32
Epoch [2/20], Step[0/8], Loss: 2.2409, Perplexity:  9.40
Epoch [3/20], Step[0/8], Loss: 1.9585, Perplexity:  7.09
Epoch [4/20], Step[0/8], Loss: 1.7507, Perplexity:  5.76
Epoch [5/20], Step[0/8], Loss: 1.6050, Perplexity:  4.98
Epoch [6/20], Step[0/8], Loss: 1.5125, Perplexity:  4.54
Epoch [7/20], Step[0/8], Loss: 1.4496, Perplexity:  4.26
Epoch [8/20], Step[0/8], Loss: 1.4057, Perplexity:  4.08
Epoch [9/20], Step[0/8], Loss: 1.3750, Perplexity:  3.95
Epoch [10/20], Step[0/8], Loss: 1.3523, Perplexity:  3.87
Epoch [11/20], Step[0/8], Loss: 1.3351, Perplexity:  3.80
Epoch [12/20], Step[0/8], Loss: 1.3221, Perplexity:  3.75
Epoch [13/20], Step[0/8], Loss: 1.3124, Perplexity:  3.72
Epoch [14/20], Step[0/8], Loss: 1.3045, Perplexity:  3.69
Epoch [15/20], Step[0/8], Loss: 1.2978, Perplexity:  3.66
Epoch [16/20], Step[0/8], Loss: 1.2920, Perplexity:  3.64
Epoch [17/20], Step[0/8], Loss: 1.2869, Perplexity:  3.62
Epoch [18/20], Step[0/8], Loss: 1.2820, Perplexity:  3.60
Epoch [19/20], Step[0/8], Loss: 1.2772, Perplexity:  3.59
Epoch [20/20], Step[0/8], Loss: 1.2724, Perplexity:  3.57
tokens= 5231
len(ids)= 5231
ids.size(0)= 5231
batch_size= 20
num_batches= 261
len(ids)= 5220
ids.shape= torch.Size([20, 261])
dictionary= {0: 'the', 1: 'little', 2: 'pig', 3: '<eos>', 4: 'every', 5: 'white', 6: 'cat', 7: 'chase', 8: 'a', 9: 'bite', 10: 'black', 11: 'dog', 12: 'love'}
vocab_size= 13
Sampled [100/1000] words and save to english_rnn.txt
Sampled [200/1000] words and save to english_rnn.txt
Sampled [300/1000] words and save to english_rnn.txt
Sampled [400/1000] words and save to english_rnn.txt
Sampled [500/1000] words and save to english_rnn.txt
Sampled [600/1000] words and save to english_rnn.txt
Sampled [700/1000] words and save to english_rnn.txt
Sampled [800/1000] words and save to english_rnn.txt
Sampled [900/1000] words and save to english_rnn.txt
Sampled [1000/1000] words and save to english_rnn.txt
```