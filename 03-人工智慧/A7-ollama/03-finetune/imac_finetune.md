

* [How to finetune Llama 3 LLM model on Macbook](https://medium.com/@elijahwongww/how-to-finetune-llama-3-model-on-macbook-4cb184e6d52e)

* [Local LLM Fine-Tuning on Mac (M1 16GB)](https://towardsdatascience.com/local-llm-fine-tuning-on-mac-m1-16gb-f59f4f598be7)
    * https://www.youtube.com/watch?v=3PIqhdRzhxE

* [Fine-tuning LLMs locally with Apple Silicon](https://heidloff.net/article/fine-tuning-llm-locally-apple-silicon-m3/)

* https://www.llama.com/docs/how-to-guides/fine-tuning/
    * https://github.com/pytorch/torchtune
    * https://pytorch.org/torchtune/main/tutorials/first_finetune_tutorial.html
    * https://github.com/pytorch/torchtune/issues/1572
    * Currently we don't support MPS out of the box, this is something that's been in the works. One of the reasons is that torchao doesn't offer binaries for MPS. You might be able to get around this by installing tochao from source @vijaygkd, before you install torchtune.

