import torch
import torch.nn.functional as F
import transformers

class GPT2LMHeadModel:
    def __init__(self, model, model_name):
        self.model = model
        self.tokenizer = transformers.GPT2Tokenizer.from_pretrained(model_name)

    @staticmethod
    def from_pretrained(model_name):
        model = transformers.GPT2LMHeadModel.from_pretrained(model_name)
        return GPT2LMHeadModel(model, model_name)

    def generate(
        self,
        input_ids,
        attention_mask=None,
        max_length=50,
        num_return_sequences=1,
        temperature=1.0,
        top_k=0,
        top_p=0.0,
        do_sample=False,
        pad_token_id=None,
        eos_token_id=None,
        no_repeat_ngram_size=0,
    ):
        """
        自定義的 generate 函數。

        參數:
        - input_ids: 輸入的 token ID 張量 (shape: [batch_size, seq_len])。
        - attention_mask: 注意力遮罩 (shape: [batch_size, seq_len])。
        - max_length: 生成的最大長度。
        - num_return_sequences: 返回的序列數量。
        - temperature: 溫度參數，控制生成的多樣性。
        - top_k: Top-k 採樣的 k 值。
        - top_p: Top-p（核採樣）的 p 值。
        - do_sample: 是否使用採樣（否則使用貪婪搜索）。
        - pad_token_id: 填充 token 的 ID。
        - eos_token_id: 結束 token 的 ID。
        - no_repeat_ngram_size: 避免重複的 n-gram 大小。
        """
        # 確保輸入是 2D 張量
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        # 初始化輸出序列
        batch_size = input_ids.size(0)
        generated = input_ids

        # 如果未提供 attention_mask，則創建一個全為 1 的遮罩
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # 如果未提供 pad_token_id，則使用 tokenizer 的 eos_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id

        # 如果未提供 eos_token_id，則使用 tokenizer 的 eos_token_id
        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id

        # 生成文本
        for _ in range(max_length - input_ids.size(1)):
            # 獲取模型的下一個 token 預測
            with torch.no_grad():
                outputs = self.model(generated, attention_mask=attention_mask)
                next_token_logits = outputs.logits[:, -1, :]

            # 溫度控制
            next_token_logits = next_token_logits / temperature

            # Top-k 過濾
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float("Inf")

            # Top-p（核採樣）過濾
            if top_p > 0.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # 移除累積概率超過 top_p 的 token
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = -float("Inf")

            # 避免重複的 n-gram
            if no_repeat_ngram_size > 0:
                for batch_idx in range(batch_size):
                    generated_sequence = generated[batch_idx].tolist()
                    for ngram in zip(*[generated_sequence[i:] for i in range(no_repeat_ngram_size)]):
                        if len(set(ngram)) == 1:  # 如果 n-gram 重複
                            next_token_logits[batch_idx, ngram[0]] = -float("Inf")

            # 計算概率分佈
            probs = F.softmax(next_token_logits, dim=-1)

            # 選擇下一個 token
            if do_sample:
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(probs, dim=-1, keepdim=True)

            # 將生成的 token 添加到序列中
            generated = torch.cat([generated, next_token], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.ones((batch_size, 1), device=attention_mask.device)], dim=-1)

            # 如果生成的是 eos_token，則停止生成
            if next_token.item() == eos_token_id:
                break

        return generated
