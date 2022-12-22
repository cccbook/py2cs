# Galois Field (有限體)

## 理論

* [用十分鐘快速掌握《數學的整體結構》](https://www.slideshare.net/ccckmit/ss-68579935)
* [維基百科:群](https://zh.wikipedia.org/wiki/%E7%BE%A4)
 * https://en.wikipedia.org/wiki/Group_(mathematics)
* [維基百科:有限體 (伽羅瓦體)](https://zh.wikipedia.org/wiki/%E6%9C%89%E9%99%90%E5%9F%9F)
  * https://en.wikipedia.org/wiki/Finite_field

## 群

* 封閉性： a * b in G
* 結合性： (a * b) * c = a * (b * c)
* 單位元素： a * I = a   # 在乘法運算中 I=1, 加法運算中 I=0
* 反元素： a * ia = I    # ia 為 a 的反元素，也是 Ｇ 的成員

具備『交換性』的群稱為『交換群』或『阿貝爾群』（Abelian group）

## GaloisField.js

```
PS D:\ccc\ccc109a\se\deno\alg\14-cryptography\GaloisField> deno run .\GaloisField.js
============================

0 + 0 = 0 mod 7
0 + 1 = 1 mod 7
0 + 2 = 2 mod 7
0 + 3 = 3 mod 7
0 + 4 = 4 mod 7
0 + 5 = 5 mod 7
0 + 6 = 6 mod 7

1 + 0 = 1 mod 7
1 + 1 = 2 mod 7
1 + 2 = 3 mod 7
1 + 3 = 4 mod 7
1 + 4 = 5 mod 7
1 + 5 = 6 mod 7
1 + 6 = 0 mod 7

2 + 0 = 2 mod 7
2 + 1 = 3 mod 7
2 + 2 = 4 mod 7
2 + 3 = 5 mod 7
2 + 4 = 6 mod 7
2 + 5 = 0 mod 7
2 + 6 = 1 mod 7

3 + 0 = 3 mod 7
3 + 1 = 4 mod 7
3 + 2 = 5 mod 7
3 + 3 = 6 mod 7
3 + 4 = 0 mod 7
3 + 5 = 1 mod 7
3 + 6 = 2 mod 7

4 + 0 = 4 mod 7
4 + 1 = 5 mod 7
4 + 2 = 6 mod 7
4 + 3 = 0 mod 7
4 + 4 = 1 mod 7
4 + 5 = 2 mod 7
4 + 6 = 3 mod 7

5 + 0 = 5 mod 7
5 + 1 = 6 mod 7
5 + 2 = 0 mod 7
5 + 3 = 1 mod 7
5 + 4 = 2 mod 7
5 + 5 = 3 mod 7
5 + 6 = 4 mod 7

6 + 0 = 6 mod 7
6 + 1 = 0 mod 7
6 + 2 = 1 mod 7
6 + 3 = 2 mod 7
6 + 4 = 3 mod 7
6 + 5 = 4 mod 7
6 + 6 = 5 mod 7
============================

0 * 0 = 0 mod 7
0 * 1 = 0 mod 7
0 * 2 = 0 mod 7
0 * 3 = 0 mod 7
0 * 4 = 0 mod 7
0 * 5 = 0 mod 7
0 * 6 = 0 mod 7

1 * 0 = 0 mod 7
1 * 1 = 1 mod 7
1 * 2 = 2 mod 7
1 * 3 = 3 mod 7
1 * 4 = 4 mod 7
1 * 5 = 5 mod 7
1 * 6 = 6 mod 7

2 * 0 = 0 mod 7
2 * 1 = 2 mod 7
2 * 2 = 4 mod 7
2 * 3 = 6 mod 7
2 * 4 = 1 mod 7
2 * 5 = 3 mod 7
2 * 6 = 5 mod 7

3 * 0 = 0 mod 7
3 * 1 = 3 mod 7
3 * 2 = 6 mod 7
3 * 3 = 2 mod 7
3 * 4 = 5 mod 7
3 * 5 = 1 mod 7
3 * 6 = 4 mod 7

4 * 0 = 0 mod 7
4 * 1 = 4 mod 7
4 * 2 = 1 mod 7
4 * 3 = 5 mod 7
4 * 4 = 2 mod 7
4 * 5 = 6 mod 7
4 * 6 = 3 mod 7

5 * 0 = 0 mod 7
5 * 1 = 5 mod 7
5 * 2 = 3 mod 7
5 * 3 = 1 mod 7
5 * 4 = 6 mod 7
5 * 5 = 4 mod 7
5 * 6 = 2 mod 7

6 * 0 = 0 mod 7
6 * 1 = 6 mod 7
6 * 2 = 5 mod 7
6 * 3 = 4 mod 7
6 * 4 = 3 mod 7
6 * 5 = 2 mod 7
6 * 6 = 1 mod 7
```