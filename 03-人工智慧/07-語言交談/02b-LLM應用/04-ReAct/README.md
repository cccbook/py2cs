# ReAct

基本上就是讓 LLM 自己規劃需要找甚麼資源，然後再去找對應的資源回來讀，反覆幾次，得到答案 ...

## 說明

* https://www.promptingguide.ai/zh/techniques/react

链式思考 (CoT) 提示显示了 LLMs 执行推理轨迹以生成涉及算术和常识推理的问题的答案的能力，以及其他任务 (Wei等人，2022)。但它因缺乏和外部世界的接触或无法更新自己的知识，而导致事实幻觉和错误传播等问题。

LLMs 以交错的方式生成 推理轨迹 和 任务特定操作 

ReAct 是一个将推理和行为与 LLMs 相结合通用的范例。ReAct 提示 LLMs 为任务生成口头推理轨迹和操作。这使得系统执行动态推理来创建、维护和调整操作计划，同时还支持与外部环境(例如，Wikipedia)的交互，以将额外信息合并到推理中。下图展示了 ReAct 的一个示例以及执行问题回答所涉及的不同步骤。

## 論文

* [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)

## 參考

* https://www.promptingguide.ai/techniques/react

