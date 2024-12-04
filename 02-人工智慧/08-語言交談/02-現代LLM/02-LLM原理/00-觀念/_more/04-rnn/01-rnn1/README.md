# language model

## 產生 train.txt

```
$ python gen_train.py > train.txt
```

## 學習 train.txt 並產生 sample.txt

輸入： train.txt

輸出： sample.txt

```
$ python main.py
Epoch [1/5], Step[0/22], Loss: 2.7744, Perplexity: 16.03
Epoch [2/5], Step[0/22], Loss: 1.4379, Perplexity:  4.21
Epoch [3/5], Step[0/22], Loss: 1.3359, Perplexity:  3.80
Epoch [4/5], Step[0/22], Loss: 1.3163, Perplexity:  3.73
Epoch [5/5], Step[0/22], Loss: 1.2976, Perplexity:  3.66
Sampled [100/1000] words and save to sample.txt
Sampled [200/1000] words and save to sample.txt
Sampled [300/1000] words and save to sample.txt
Sampled [400/1000] words and save to sample.txt
Sampled [500/1000] words and save to sample.txt
Sampled [600/1000] words and save to sample.txt
Sampled [700/1000] words and save to sample.txt
Sampled [800/1000] words and save to sample.txt
Sampled [900/1000] words and save to sample.txt
Sampled [1000/1000] words and save to sample.txt
```


## 參考文獻

* https://github.com/pyliaorachel/resurrecting-the-dead-chinese.git

* [NLP 笔记 - Language models and smoothing](http://www.shuang0420.com/2017/02/24/NLP%20%E7%AC%94%E8%AE%B0%20-%20Language%20models%20and%20smoothing/)

Unknown word

对 unknown word 的处理，一般我们建一个固定大小的 lexicon(比如说语料库里 frequency>5 的单词)，再新建一个 token <UNK>，不在 lexicon 里的 token (也就是 frequency<5 的单词)都编译成 <UNK>，然后把 <UNK> 当做普通单词处理。