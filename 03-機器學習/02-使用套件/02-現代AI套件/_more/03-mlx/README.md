# mlx

注意： pytorch 是有支援 mac GPU 的

* https://pytorch.org/blog/introducing-accelerated-pytorch-training-on-mac/

* https://chatgpt.com/c/670c7150-9a0c-8012-b9e2-a27ec87da36a
    * Metal 是 Apple 開發的一種低階圖形處理 API，類似於 Vulkan 或 DirectX 12。Metal 允許開發者在 iOS、macOS 和 Apple Silicon 上直接訪問 GPU，以實現高性能的圖形和計算處理

* https://chatgpt.com/c/670c7a54-8d1c-8012-8445-19f5122d7bfc
    * Metal Performance Shaders (MPS) with PyTorch:PyTorch 最近版本開始對 Apple Silicon 提供支援，並利用 Apple 的 Metal API 來加速 GPU 運算。你可以使用 PyTorch 的 torch.device("mps") 來在 Apple Silicon 上啟用 GPU 加速。
    * https://pytorch.org/docs/stable/notes/mps.html

## 官方

* https://ml-explore.github.io/mlx/build/html/
* https://github.com/ml-explore/mlx
    * https://github.com/ml-explore/mlx-examples/

## 參考
* [A simple guide to local LLM fine-tuning on a Mac with MLX](https://apeatling.com/articles/simple-guide-to-local-llm-fine-tuning-on-a-mac-with-mlx/)
    1. [Setting up your environment.](https://apeatling.com/articles/part-1-setting-up-your-environment/)
    2. [Building your training data for fine-tuning](https://apeatling.com/articles/part-2-building-your-training-data-for-fine-tuning/)
    3. [Fine-tuning your LLM using the MLX framework](https://apeatling.com/articles/part-3-fine-tuning-your-llm-using-the-mlx-framework/)
    4. [Testing and interacting with your fine-tuned LLM](https://apeatling.com/articles/part-4-testing-and-interacting-with-your-fine-tuned-llm/)



* https://github.com/ml-explore/mlx-examples

