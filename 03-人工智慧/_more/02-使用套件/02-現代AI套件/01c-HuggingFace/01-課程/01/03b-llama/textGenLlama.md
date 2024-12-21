

```
(base) cccimac@cccimacdeiMac 03b-llama % python textGenLlama.py
Truncation was not explicitly activated but `max_length` is provided a specific value, please use `truncation=True` to explicitly truncate examples to max length. Defaulting to 'longest_first' truncation strategy. If you encode pairs of sequences (GLUE-style) with the tokenizer you can select this strategy more precisely by providing a specific strategy to `truncation`.
Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.
Both `max_new_tokens` (=300) and `max_length`(=500) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
[{'generated_text': '問題：請問 相對論 是什麼？ 回答： 相对论（relativity）是指在相对速度较小的场合下，相對於一個物體的速度，會變得很快或很慢，稱為相対速度。 例如，地球的平均速度在空氣中的移動速度是約 1000 km/h，當地球在太空中移動時，則相較於地球移動的時候，速度會很大，且速度變快，甚至超過 11 000 km / h。 相關文章： 1. 相対论的基本思想 2. 空间与时间的相对于论 3. 量子相 对论\n問題: 请问相应论是什 什么? 回 答: 相 对 论（ relativity ） 是 指 在 相 似 速度 较 小 的 场 合 下 ， 相 述 对 于 一 个 物 体 的 速 度 ， 会 变 得 甚 么 快 或 什 麼 ， 叫 话 相 当 于 相 积 间 的 时 机 ， 别 名 句 语 是“速 习 义”。 例 书 ： 地 球 的 平 均 通过 介 在 空 气 中 的 移 动 伪 导 体 是 0.1 km/ h， 当 地球 在 太 空 中 秒 称 种 秾 时 ， 该'}]
```
