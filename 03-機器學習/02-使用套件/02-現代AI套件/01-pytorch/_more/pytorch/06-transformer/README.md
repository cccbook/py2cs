# Attention / Transformer / BERT 

## Transformer (Attention)

* https://d2l.ai/chapter_attention-mechanisms/index.html (超讚)

* [The Annotated Transformer (Attention is All You Need)](http://nlp.seas.harvard.edu/annotated-transformer/) (超讚)
    * 中文版 -- https://blog.csdn.net/qq_56591814/article/details/120278245
    * code -- https://github.com/harvardnlp/annotated-transformer/

Transformer 透過 Attention 機制取代 seq2seq 中的循環神經網路。

## BERT

BERT 則是雙向的 Transformer ，透過克漏字 (Masked Language Model) 和下一句預測 (Next Sentence Model)，來達成自然語言的訓練。

克漏字: Google AI 研究人员随机掩盖了每个序列中 15% 的单词

## 科普

* [一文理解 Transformer 的工作原理](https://www.infoq.cn/article/qbloqm0rf*sv6v0jmulf), Prateek Joshi 刘志勇
* Programming with Data 柯頌竹
    * [28. 注意力機制（Attention mechanism）](https://medium.com/programming-with-data/28-%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%A9%9F%E5%88%B6-attention-mechanism-f3937f289023)
    * [29. Transformer](https://medium.com/programming-with-data/29-transformer-2ac4b5f31072)

## 程式

* https://github.com/fkodom/transformer-from-scratch
    * [Transformers from Scratch in PyTorch](https://medium.com/the-dl/transformers-from-scratch-in-pytorch-8777e346ca51)
* 李弘毅習題
    * https://github.com/Fafa-DL/Lhy_Machine_Learning/tree/main/2021%20ML

## 論文

* Attention Is All You Need
    * https://arxiv.org/abs/1706.03762
    * https://paperswithcode.com/paper/attention-is-all-you-need
    * [中文版](https://blog.csdn.net/baidu_41617231/article/details/118079142)

* https://pytorch.org/hub/huggingface_pytorch-transformers/
* https://huggingface.co/docs/transformers/task_summary

* https://www.facebook.com/yuenhsien.tseng/posts/10223425030358249

知識的擴散  出了什麼問題？
#新的計算機概論
去年（2021年）審查一本澳洲知名大學的博士學位論文，其運用了CNN、RNN等深度神經網路來解問題，但是卻沒有用到效果可能更好的 Transformer 網路。2017年Transformer被提出來，2018年已經非常火紅了，但2021年的博士論文沒有運用到，讓我覺得難以想像。
今年（2022）剛剛審完一份某國立大學人工智慧研究所新進研究人員的科技部計畫，居然也還僅止於使用 LSTM、GRU等RNN網路，都沒有提到 Transformer、Graph Neural Networks（GNN）等對其問題可能效果較佳的方法，再次讓我震撼，心想到底為何會如此。
2017年Transformer的出現是自然語言處理的分水嶺，現在頂尖會議的論文，或是期刊論文，若能夠用 Transformer 的場合而不用的話，幾乎沒有被接受的可能。可以說2017年以前的技術，算是農耕時代（甚至是石器時代）的老舊技術了，是還可以用，但在大家都在比較系統成效（effectiveness），而很少比較系統效率（efficiency）的情況下，是很難回得去只用 RNN、LSTM、GRU 的技術了。Transformer甚至被運用到影像、語音的處理，而讓不同模態的資料，可以用同一個神經網路架構來處理，甚至可以跨模態處理，而可以用語言表達（輸入），讓電腦作畫（輸出），或是反過來，讓電腦針對輸入的圖片，產生簡要的圖說。
在解決問題上，以往還會在乎領域知識的融入，但使用Transformer，幾乎只要有訓練資料即可。
可以說，Transformer 以及其相關的技術，已經是新的計算機概論了。
所以，為何這些年輕學者，還沒有用、甚至沒有提到？他們的老師會不知道？他們的週遭同學、伙伴會不知道？沒有參加國際會議得知最新技術概況？網路上唾手可得的教材、視頻、範例程式，不足以教會這些年輕學者？
還在納悶中...

* https://github.com/huggingface/transformers

You should install Transformers in a virtual environment. If you're unfamiliar with Python virtual environments, check out the user guide.

## 參考文獻
* [一文看懂 Attention（本质原理+3大优点+5大类型）](https://easyaitech.medium.com/%E4%B8%80%E6%96%87%E7%9C%8B%E6%87%82-attention-%E6%9C%AC%E8%B4%A8%E5%8E%9F%E7%90%86-3%E5%A4%A7%E4%BC%98%E7%82%B9-5%E5%A4%A7%E7%B1%BB%E5%9E%8B-e4fbe4b6d030)
* [Transformer 代码完全解读！](https://bbs.cvmart.net/articles/5563)
* 莫凡 NLP -- https://github.com/MorvanZhou/NLP-Tutorials
    * https://mofanpy.com/tutorials/machine-learning/nlp/
    * [Transformer 自注意语言模型 #5.4 (莫烦Python NLP 自然语言处理教学)](https://www.youtube.com/watch?v=KBl3ZTeLqdo)
* Programming with Data 柯頌竹
    * [28. 注意力機制（Attention mechanism）](https://medium.com/programming-with-data/28-%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%A9%9F%E5%88%B6-attention-mechanism-f3937f289023)
    * [29. Transformer](https://medium.com/programming-with-data/29-transformer-2ac4b5f31072)
    * [30. Transformer Performance 評估與應用討論](https://medium.com/programming-with-data/30-transformer-performance-%E8%A9%95%E4%BC%B0%E8%88%87%E6%87%89%E7%94%A8%E8%A8%8E%E8%AB%96-913fa46690da)
    * [31. ELMo (Embeddings from Language Models 嵌入式語言模型)](https://medium.com/programming-with-data/31-elmo-embeddings-from-language-models-%E5%B5%8C%E5%85%A5%E5%BC%8F%E8%AA%9E%E8%A8%80%E6%A8%A1%E5%9E%8B-c59937da83af)
    * [32. Transformer + 預訓練 : 集大成的 Bert 模型](https://medium.com/programming-with-data/32-transformer-%E9%A0%90%E8%A8%93%E7%B7%B4-%E9%9B%86%E5%A4%A7%E6%88%90%E7%9A%84-bert-%E6%A8%A1%E5%9E%8B-c928530f6db8)
    * [33. 輕量化 Bert 應用範例](https://medium.com/programming-with-data/33-%E8%BC%95%E9%87%8F%E5%8C%96-bert-%E6%87%89%E7%94%A8%E7%AF%84%E4%BE%8B-aa61b974c0f3)
