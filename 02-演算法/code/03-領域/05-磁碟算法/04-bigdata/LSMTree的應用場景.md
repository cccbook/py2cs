## LSM Tree

本文擷取自 [LSM Tree 介紹](https://medium.com/@thegiive/lsm-tree-%E4%BB%8B%E7%B4%B9-3dc32873fa66)

## 應用場景

* [Write-ahead logging](https://en.wikipedia.org/wiki/Write-ahead_logging)
* [预写式日志](https://zh.wikipedia.org/wiki/%E9%A2%84%E5%86%99%E5%BC%8F%E6%97%A5%E5%BF%97)

LSM (Log Structured-Merge Tree) 第一次發表是來自 Google BigTable 論文，他出現是為了大數據 OLAP 場景 heavy write throughput 可以犧牲 read 的速度。

## 方法

1. LSM Write 一開始會先把 data append 到 WAL File 裡面
2. 在 memory 裡面，會用 AVL Tree or 其他 Tree 等 sorted tree 方式來 index data，這個叫做 memtable。
3. memtable 經過一段時間，或是達成某個 criteria 之後( size > 某個值）， batch 會把 memtable 的東西實體化成 SSTable file 。這裡依舊是 sequential write operation。
4. 每個 SSTable 都是 immutable , 時間久了自然就會有很多 SSTable File, 同一個 key 的值可能因為 update 次數很多，同一個 key 的多個版本的 data 散落在多個 SSTable File 裡面。這裡就需要 backend process 定期對於 SS Table 進行合併，這就叫做 compact 。
5. READ: LSM read 就比較麻煩，LSM 必須從 memtable 尋找是否有這個 key 的數據，如果沒有，就一個一個從 SS Table 來尋找。這就是所說的犧牲了 read 性能。

![B-Tree v.s. LSM-Tree](https://twitter.com/alexxubyte/status/1583119489318518786)