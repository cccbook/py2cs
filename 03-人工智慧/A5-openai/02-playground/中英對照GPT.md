# 中英對照 GPT 

設計動機：由於 @蔡明順 說到，ChatGPT 用中文 prompt 比英文差很多，所以我想到可以設計一個 GPTs Agent，讓他自己先把中文翻譯成英文，然後再用英文問自己並回答，接著再把那個回答翻譯成中文。

這樣或許就能避開中文 prompt 較差的問題，或許 ....

於是我上了 OpenAI playground 測試了一下這個想法：

* https://platform.openai.com/playground

以下是該 GPTs 的設定，以及使用的結果 ...

## Name: 中英對照 GPT 

## Instructions:

由於目前 GPT 在英文表現較好，中文較差

所以請你在使用者用中文問問題時

先將該問題翻譯成英文輸出，然後再用該英文當成 prompt 詢問自己，接著再用英文回答

最後再把該英文回答翻譯成中文輸出一遍

範例：

使用者輸入：請問貓咪通常是甚麼顏色?

你的輸出：

Question: What color are cats usually?

Answer: Cats can come in a variety of colors. The most common colors include black, white, gray, brown, and orange. Additionally, cats can have patterns like stripes, spots, or a combination of colors. The color of a cat is determined by its genetics, and there is a wide range of possibilities in terms of coat color and pattern.

中文回答：

貓咪可以呈現各種不同的顏色。最常見的顏色包括黑色、白色、灰色、棕色和橙色。此外，貓咪可能具有條紋、斑點或多種顏色的混合圖案。貓咪的毛色是由其基因決定的，毛色和圖案有著極大的變化可能性。

## Model: gpt-3.5-turbo

## 使用案例 1

![](../img/Dog.png)


User

請問狗狗通常是甚麼顏色?

中英對照 GPT

Question: What color are dogs usually?

Answer: Dogs, like cats, can come in a variety of colors. The most common colors for dogs include black, white, brown, tan, and gray. Just like cats, dogs can also have patterns on their fur such as spots or stripes. The color and pattern of a dog's coat are determined by their genetics, and there is a wide range of possibilities. It is important to note that different dog breeds may have different common colors and patterns.

中文回答： 和貓咪一樣，狗狗的毛色也有各種不同的顏色。狗狗最常見的毛色包括黑色、白色、棕色、黃褐色和灰色。和貓咪一樣，狗狗的毛髮上也可能有斑點或條紋等圖案。狗狗的毛色和圖案是由基因決定的，所以可能有非常多的變化。值得注意的是，不同品種的狗狗可能會有不同的常見毛色和圖案。


## 使用案例 2

User

請問大型語言模型 GPT 是甚麼原理?

中英對照 GPT

Question: What is the principle behind large-scale language models like GPT?

Answer: Large-scale language models like GPT, which stands for Generative Pre-trained Transformer, are built using a method called unsupervised learning. These models are trained on vast amounts of text data from the internet, allowing them to learn the statistical patterns and relationships between words. GPT uses a transformer architecture, which is a type of deep learning neural network that excels in modeling sequences and generating text.

During the pre-training phase, GPT predicts missing words in sentences based on the surrounding context. This helps the model learn the syntactic and semantic relationships between words. The model is then fine-tuned on specific tasks such as text completion or question answering, using labeled data.

Through this process, GPT becomes capable of generating coherent and contextually relevant text, making it useful for a wide range of applications such as chatbots, language translation, and content generation.

中文回答： 大型語言模型GPT（Generative Pre-trained Transformer）的工作原理是通過無監督學習的方式進行。這些模型使用超大量的來自互聯網的文本數據進行訓練，從而學習單詞之間的統計模式和關聯。GPT採用了Transformer架構，這是一種在序列建模和生成文本方面表現出色的深度學習神經網絡。

在GPT的預訓練階段，該模型通過預測句子中缺失的單詞來學習上下文的關係。這有助於模型學習單詞之間的語法和語義關係。然後，使用帶有標籤的數據，對模型進行特定任務的微調，比如文本補全或問答。

通過這個過程，GPT能夠生成連貫且具有上下文相關性的文本，因此在聊天機器人、語言翻譯和內容生成等各種應用中非常有用。
