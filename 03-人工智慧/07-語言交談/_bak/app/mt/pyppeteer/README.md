# python 呼叫『騰訊翻譯』

原理：使用 Pyppeteer 用 headless browser 偽裝瀏覽器丟請求給『騰訊翻譯』

## 執行

```
(env) mac020:pyppeteer mac020$ python qqmt1.py
======== 原文 ========
The number of micro operations is minimized without impacting the quality of the generated code much. For example, instead of generating every possible move between every 32 PowerPC registers, we just generate moves to and from a few temporary registers. These registers T0, T1, T2 are typically stored in host registers by using the GCC static register variable extension.
======== 譯文 ========
微操作的数量被最小化，而不会对生成的代码的质量产生太大影响。例如，我们不是在每32个PowerPC寄存器之间生成每个可能的移动，而是只生成几个临时寄存器之间的移动。这些寄存器T0、T1、T2通常通过使用GCC静态寄存器变量扩展存储在主机寄存器中。
```

## 參考文獻

* https://miyakogi.github.io/pyppeteer/
* [pyppeteer(python版puppeteer)基本使用](https://www.cnblogs.com/baihuitestsoftware/p/10531462.html)