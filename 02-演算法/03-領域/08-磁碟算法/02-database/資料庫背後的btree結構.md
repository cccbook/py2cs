# 資料庫背後的 B-Tree結構

* [B樹](https://zh.wikipedia.org/zh-tw/B%E6%A0%91)
    * [B-Tree](https://en.wikipedia.org/wiki/B-tree)
* [B+树](https://zh.wikipedia.org/wiki/B%2B%E6%A0%91)
    * [B+ Tree](https://en.wikipedia.org/wiki/B%2B_tree)

## B-Tree

Invented by	Rudolf Bayer, Edward M. McCreight

多元樹 (例如分支 100 以上，最好能一個節點放入一個硬碟區塊)，適合用來存放硬碟資料，降低硬碟讀取次數。

## B+ Tree

A B+ tree can be viewed as a B-tree in which each node contains only keys (not key–value pairs), and to which an additional level is added at the bottom with linked leaves.
