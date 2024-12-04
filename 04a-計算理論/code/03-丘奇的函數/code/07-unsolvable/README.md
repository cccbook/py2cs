## Lambda Calculus 中的不可計算問題

* [λ演算](https://zh.wikipedia.org/wiki/%CE%9B%E6%BC%94%E7%AE%97)
* [Church, A. An unsolvable problem of elementary number theory.](https://www.ics.uci.edu/~lopes/teaching/inf212W12/readings/church.pdf)
    * https://www.jstor.org/stable/2371045
    * https://archive.org/details/sim_american-journal-of-mathematics_1936-04_58_2/page/345/mode/2up
    * https://www.ics.uci.edu/~lopes/teaching/inf212W12/readings/church.pdf

lambda表达式之间的等价性，无法找到某个通用的函数来判定。

在1936年邱奇利用λ演算給出了對於判定性問題（Entscheidungsproblem） 的否定：關於兩個lambda運算式是否等價的命題，無法由一個「通用的演算法」判斷，這是不可判定效能夠證明的頭一個問題，甚至還在停机问题之先。

THEOREM XVIII, There is no recursive function of a formula C, whose value is 2 or 1 according as C has a normal form or not.

THEOREM XIX. There is no recursive function of two formulas A and B, whose value is 2 or 1 according as A conv B or not.
