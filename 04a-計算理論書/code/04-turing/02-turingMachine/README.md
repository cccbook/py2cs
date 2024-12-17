# Turing Machine


* [圖靈機](https://zh.wikipedia.org/zh-tw/%E5%9B%BE%E7%81%B5%E6%9C%BA)
* 測試的 anbncn 圖靈機來源
    * https://www.cs.odu.edu/~zeil/cs390/latest/Public/turing-jflap/index.html

## TuringMachine.py

```
$ python TuringMachine.py
: True
ab: False
abc: True
aabbc: False
aabbcc: True
```

## 狀態機

* 狀態機繪製工具 -- https://edotor.net/

![](./img/TuringMachine_anbncn.png)

上圖的原始碼

```
digraph TuringMachine_anbncn {
	rankdir=LR;
	size="8,5"

	node [shape = doublecircle]; 0 3;
	node [shape = circle];

	0 -> 1 [ label = "a/_,R" ];
	1 -> 1 [ label = "a/a,R" ];
	1 -> 1 [ label = "x/x,R" ];
	1 -> 2 [ label = "b/x,R" ];
	2 -> 2 [ label = "x/x,R" ];
	2 -> 2 [ label = "b/b,R" ];
	2 -> 5 [ label = "c/x,L" ];
	5 -> 5 [ label = "x/x,L" ];
	5 -> 5 [ label = "a/a,L" ];
	5 -> 0 [ label = "_/_,R" ];
	0 -> 4 [ label = "x/x,R" ];
	0 -> 3 [ label = "_/_,L" ];
	4 -> 4 [ label = "x/x,R" ];
	4 -> 3 [ label = "_/_,L" ];
}

```
