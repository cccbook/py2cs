# Attention is All You Need

本文為閱讀下文之後的心得

* [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/)
* [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

## Transformer 架構

![](./img/Transformer.png)

1. Encoder-Decoder 模型
2. 最小化 (input, output) 兩者輸入後得到的 Loss ，期望當 input 輸入後，當 output 輸入時， loss 最小。

## Attention 的設計

![](./img/Attention0.png)


An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.

注意力： 

1. 功能為 map(q, k=>v)
2. q 和 k 若相容，則提取值大，若不相容，則提取值小。(有點像卷積神經網路的遮罩)
