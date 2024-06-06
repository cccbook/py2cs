
* https://github.com/pytorch/tutorials/blob/main/beginner_source/transformer_tutorial.py
    * https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    * https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/9cf2d4ead514e661e20d2070c9bf7324/transformer_tutorial.ipynb#scrollTo=Up6vzGF0X1ya
    * https://github.com/pytorch/tutorials/blob/main/beginner_source/transformer_tutorial.py
    * https://blog.floydhub.com/the-transformer-in-pytorch/

注意：這篇訓練的是 nn.TransformerEncoder model ，而非完整的 Transformer Encoder + Decoder.

所以不訓練 Q&A ，單純訓練語言模型。

In this tutorial, we train a nn.TransformerEncoder model on a language modeling task. The language modeling task is to assign a probability for the likelihood of a given word (or a sequence of words) to follow a sequence of words. A sequence of tokens are passed to the embedding layer first, followed by a positional encoding layer to account for the order of the word (see the next paragraph for more details).

下面這篇訓練翻譯，才是完整的 transformer 。

* https://github.com/pytorch/tutorials/blob/main/beginner_source/translation_transformer.py

