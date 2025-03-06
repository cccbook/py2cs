# 多變數積分

用 ChatGPT 學

* 

```
$ python int2var.py
積分結果: 0.6666666666666669
誤差估計: 1.4754810833321613e-14

ccckmit@asus MINGW64 /d/ccc/ccc112a/py2cs/01-數學/04-微積分/02-多變數微積分/chat (master)
$ python int5var.py
積分結果: 1.666666666666667
誤差估計: 4.7917013087034885e-14
```


## mc_int


User
    更高維的積分，難以用黎曼積分去計算的，可以怎麼估計積分結果

ChatGPT

    在更高維度的積分問題中，使用黎曼積分可能會變得非常困難或不切實際。在這種情況下，你可能會考慮使用Monte Carlo積分或Quasi-Monte Carlo積分等蒙地卡羅方法。

* https://chat.openai.com/share/b7bf0705-1d80-4df2-b46d-cbb30e3ddcf7

```
$ python mc_int1.py
估算的五變數函數積分值： 1.6666781946985034

$ python mc_int2.py
估算的六變數函數積分值： 2.134856902791531
```