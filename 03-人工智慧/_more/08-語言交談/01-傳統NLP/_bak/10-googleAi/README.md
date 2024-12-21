# Google AI

* [如何使用 Google 的 Gemini 模型 API？(基礎教學，附上 Python 範例程式)](https://blog.jiatool.com/posts/gemini_api/)

* https://ai.google.dev/
    * https://cloud.google.com/generative-ai-studio?hl=en

```py
model = genai.GenerativeModel(model_name="gemini-pro-vision")

response = model.generate_content(["What’s in this photo?", img])
```

* 

```js
const model = genAI.getGenerativeModel({ model: "gemini-pro-vision"});

const result = await model.generateContent([
  "What’s in this photo?",
  {inlineData: {data: imgDataInBase64, mimeType: 'image/png'}}
]);
```

