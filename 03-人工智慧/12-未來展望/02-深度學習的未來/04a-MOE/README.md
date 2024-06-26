

* https://www.facebook.com/allanyiin/posts/pfbid02FyU2t5ZLzMayTS8Criym4xqghsGnjvcvDt5y7qq3SL8TH6hNk4N95J5MMdqLM7Jyl
    * 尹相志: 事實證明gpt4o就只是gpt3.5以及gpt4的MOE


* https://www.ithome.com/0/738/580.htm

![](https://img.ithome.com/newsuploadfiles/2023/12/72149066-7351-4886-86f9-a4e93e3386a0.png)

```
简单来说，MoE 是一种神经网络架构设计，在 Transformer 模块中集成了专家 / 模型层。

当数据流经 MoE 层时，每个输入 token 都会动态路由到专家子模型进行处理。当每个专家专门从事特定任务时，这种方法可以实现更高效的计算并获得更好的结果。

MoE 最关键的组件：

- 专家（Expert）：MoE 层由许多专家、小型 MLP 或复杂的 LLM（如 Mistral 7B）组成。

- 路由器（Router）：路由器确定将哪些输入 token 分配给哪些专家。

路由策略有两种：token 选择路由器或路由器选择 token。

路由器使用 softmax 门控函数通过专家或 token 对概率分布进行建模，并选择前 k 个。

MoE 能够带来的好处：

- 每个专家都可以专门处理不同的任务或数据的不同部分。

- MoE 构架能向 LLM 添加可学习参数，而不增加推理成本。

- 可以利用稀疏矩阵的高效计算。

- 并行计算所有专家层，以有效利用 GPU 的并行能力。

- 帮助有效地扩展模型并减少训练时间。以更低的计算成本获得更好的结果！
```
