# ChatGPT 的技術來源

* https://chat.openai.com/chat


## 陳縕儂 

* [OpenAI InstructGPT 從人類回饋中學習 ChatGPT 的前身](https://www.youtube.com/watch?v=ORHv8yKAV2Q)

1. 有 reward model ，並且可以根據人類對 GPT 產生回答的排序進行 reword 評分調整。
2. PPO : Proximal Policy Optimization
    * PPO-ptx: + pretrain 避免離 pretrain 太遠

* [OpenAI ChatGPT 驚驗眾人的對話互動式AI](https://www.youtube.com/watch?v=TnGPmlONfI8)

## 李宏毅

* [Chat GPT (可能)是怎麼煉成的 - GPT 社會化的過程](https://www.youtube.com/watch?v=e0aKI2GGZNg)

## 英文影片

* [OpenAI's InstructGPT: Aligning Language Models with Human Intent](https://www.youtube.com/watch?v=QGpaBWOaHQI)
    * 這個影片更詳細的說明了 reword model 的原理

## 陳鍾誠

ChatGPT 這個程式並非憑空出現的！

2018年的圖靈獎頒給了 Bengio/Hinton/LeCun 等三人，以表彰他們在神經網路深度學習領域的貢獻。

(圖靈獎從來沒頒給神經網路領域的人，2018 年是第一次，也是到目前為止唯一的一次 ...)

1. Bengio 是 RNN 循環神經網路的發明人
2. LeCun 是 CNN 卷積神經網路的發明人
3. Hinton 是反傳遞演算法和受限波茲曼機 CD-K 算法的發明人

然後因為 Hinton 的學生用 GPU 去訓練神經網路導致 2011 年影像辨識程式的能力突飛猛進，後來 Bangio 的學生 Ian Goodfellow 發明 GAN 而讓影像合成有重大突破。

特斯拉的前AI總監 Karpathy 在 2015 年博士生時期，寫了數篇網誌，讓我看懂了《梯度下降法/反傳遞演算法/CNN/RNN》等技術

1. 反傳遞: Hacker's guide to Neural Networks
2. RNN : The Unreasonable Effectiveness of Recurrent Neural Networks
3. CNN : ConvNetJS 程式與網站

然後到了 2017 年，Google 以一篇

> Attention Is All You Need 

改良了 RNN 的缺陷，提出 Transformer 這樣的模型讓整個技術有了重大的突破。

接著 Google 在 2018 年改良 Transformer 為 BERT 模型，讓 Transformer 技術大放異彩

然後 openai 也開始發展出 GPT 1,2,3 ，用更大規模的 Transformer，更強調生成而非理解，於是終於有了今天 ChatGPT 的誕生。

這些論文與技術，我會整理在本文的回應連結中！

(不過像 GPT 這樣的大型技術，已經不是我所能碰的了，兩千億個參數的 Transformer 神經網路，加上大量的網路語料和標記庫，是需要資金、技術和人力的結合才能做的，像我這種只有一台小電腦的人，基本上只能用別人訓練好的模型了 ....)

## 重要資源與文獻

1. 線上電子書 -- https://d2l.ai/
2. 程式
    * CNN -- https://cs.stanford.edu/people/karpathy/convnetjs/
    * Transformer -- https://huggingface.co/docs/transformers/task_summary
    * 反傳遞 -- https://github.com/karpathy/micrograd
    * [Awesome ChatGPT](https://github.com/Kamigami55/awesome-chatgpt/blob/main/README-zh-TW.md)
3. Hinton (Back Propagation): [Learning representations by back-propagating errors (Paper)](https://www.nature.com/articles/323533a0)
4. LeNet (CNN): [GradientBased Learning Applied to Document Recognition (Paper)](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)
5. Bangio (RNN): [A Neural Probabilistic Language Model (Paper)](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
6. Hinton (CNN, ImageNet): [ImageNet Classification with Deep Convolutional Neural Networks](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
7. [Hacker's guide to Neural Networks (BLOG)](http://karpathy.github.io/neuralnets/)
8. [The Unreasonable Effectiveness of Recurrent Neural Networks (BLOG)](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
9. [Attention Is All You Need (Paper)](https://arxiv.org/abs/1706.03762)
10. [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)
11. GPT1 [Improving Language Understanding by Generative Pre-Training (Paper)](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
12. GPT2 [Language Models are Unsupervised Multitask Learners (Paper)](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
13. GPT3 -- [Language Models are Few-Shot Learners (Paper)](https://arxiv.org/pdf/2005.14165.pdf)
14. 在台灣要跑 GPT ，看來只能用台灣衫了 (但要申請，可能會花些時間)
    * [台灣杉三號 (TAIWANIA 3)](https://www.nchc.org.tw/Page?itemid=2&mid=4)
15. [shaoeChen 的 HackMD 筆記 -- 李宏毅課程: ELMO, BERT, GPT](https://hackmd.io/@shaoeChen/Bky0Cnx7L)
