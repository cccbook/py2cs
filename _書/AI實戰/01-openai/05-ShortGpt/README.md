# ShortGPT

```
$ shortgpt.sh
Welcome user to shortgpt. You may use $key for short
{
  "tw": "繁體中文",
  "en": "English",
  "md": "Markdown+LaTex,  add space before and after $..$"
}

command> chat Hello
========question=======
Hello
========response=======
Hello! How may I assist you today?

command> shell ls
01.md  GPT.md    ItoIntegration.md      stochasticCalculus_tw.md
a1.md  hello.md  stochasticCalculus.md  test.md

command> history
0:chat Hello
1:shell ls
2:history

command> fchat hello.md 你好
========question=======
你好
========response=======
Response will write to file:hello.md

command> quit
```
