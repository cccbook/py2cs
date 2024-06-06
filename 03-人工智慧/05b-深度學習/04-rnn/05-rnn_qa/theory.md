# 用 RNN 做 seq2seq 的缺點

1. RNN inputs 與 outputs 相同，但以翻譯而言，兩者應該不同， RNN 用 inputs 與 outputs 兩集合的聯集來做預測。
2. 