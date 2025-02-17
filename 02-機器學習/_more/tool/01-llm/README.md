

https://llm.datasette.io/en/stable/index.html

Install llm using pipx

    pipx install llm
​
Install the Ollama plugin

    llm install llm-ollama
​
Now you can use Ollama models.

    llm -m llama3.2-vision:90b "Describe the image. Respond in Traditional Chinese language" -a xxx.jpg