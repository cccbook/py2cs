# language model

## 產生 train.txt

```bash
$ gen.sh
```

## 學習 exp_train.txt 並產生 exp_sample.txt

training

```
$ python main.py exp train
tokens= 14388
len(ids)= 14388
ids.size(0)= 14388
batch_size= 20
num_batches= 719
len(ids)= 14380
ids.shape= torch.Size([20, 719])
vocab_size= 16
Epoch [1/3], Step[0/23], Loss: 2.7705, Perplexity: 15.97
Epoch [2/3], Step[0/23], Loss: 1.4124, Perplexity:  4.11
Epoch [3/3], Step[0/23], Loss: 1.3985, Perplexity:  4.05
```

testing

```
$ python main.py exp test
tokens= 14388
len(ids)= 14388
ids.size(0)= 14388
batch_size= 20
num_batches= 719
len(ids)= 14380
ids.shape= torch.Size([20, 719])
vocab_size= 16
Sampled [100/1000] words and save to exp_sample.txt
Sampled [200/1000] words and save to exp_sample.txt
Sampled [300/1000] words and save to exp_sample.txt
Sampled [400/1000] words and save to exp_sample.txt
Sampled [500/1000] words and save to exp_sample.txt
Sampled [600/1000] words and save to exp_sample.txt
Sampled [700/1000] words and save to exp_sample.txt
Sampled [800/1000] words and save to exp_sample.txt
Sampled [900/1000] words and save to exp_sample.txt
Sampled [1000/1000] words and save to exp_sample.txt
```

## 參考文獻

* https://github.com/pyliaorachel/resurrecting-the-dead-chinese.git

* [NLP 笔记 - Language models and smoothing](http://www.shuang0420.com/2017/02/24/NLP%20%E7%AC%94%E8%AE%B0%20-%20Language%20models%20and%20smoothing/)

Unknown word

对 unknown word 的处理，一般我们建一个固定大小的 lexicon(比如说语料库里 frequency>5 的单词)，再新建一个 token <UNK>，不在 lexicon 里的 token (也就是 frequency<5 的单词)都编译成 <UNK>，然后把 <UNK> 当做普通单词处理。