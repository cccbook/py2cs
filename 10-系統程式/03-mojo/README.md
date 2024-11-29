# mojo 想做什麼？

本資料夾程式來源 -- https://docs.modular.com/max/tutorials/magic

目標：很像 Python 效能卻可以和 C/C++/Golang 匹敵的語言。

問題是， Python 已經很穩固了， mojo 也沒辦法完全編譯 Python 那些套件，那要怎麼吸引使用者，甚至吃下 Python 的地盤呢？

方法是

1. 透過 magic cli 介面，可以創建 python/mojo 的專案，並且融合二者的編譯，執行與互相呼叫。
2. mojo 語言是 python 的 SuperSet, 和 Python 相容度高，原本就比較容易相互呼叫。
3. 推出 max 神經網路介面，可以將 tensorflow/pytorch ... 等各種框架，透過 MAX engine (compiler+runtime) 去轉換後，在 CPU/GPU/TPU 上執行。

* mojo: https://docs.modular.com/mojo/manual/
* max: https://github.com/modularml/max/tree/main
* magic: https://docs.modular.com/max/tutorials/magic/
    * https://docs.modular.com/max/tutorials/magic
    * 和 fastapi 融合 -- https://docs.modular.com/max/tutorials/magic#step-4-add-pypi-dependencies

## 參考

* https://www.facebook.com/ccckmit/posts/pfbid031NsSDojLZJ6hjmnPnPgZQh6Rn8ZX31dF6b2XyLn95ypV38cMBYWbVfdcrv8qs4ECl

* [mojo如何吃下Python地盤](https://chatgpt.com/c/67468030-0ef0-8012-8c84-3eb02d158877)

我終於知道 mojo 語言想幹嘛了

目前其實 mojo 的策略不是取代 python

而是成為『很像 python 的快速語言』

然後透過 magic 這個指令，讓 mojo/python 可以很容易的混再一起使用

透過這樣，讓 python把需要快速執行的程式用 mojo 來寫

融合久了，才有機會讓 mojo 成為 python 的取代者
