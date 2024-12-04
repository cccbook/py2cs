
ChatGPT: https://chatgpt.com/c/672dd2ce-a47c-8012-a56b-341adef952f2

* ccc: 強化學習 VPG (vanilla Policy Gradient) 簡單策略梯度算法，為何要擴充為 有 Baseline 的版本， Baseline 有何用途？

結論摘要: Baseline 的選擇很重要。最常見的做法是使用 狀態價值函數 V(s_t)，因為它估計了該狀態下後續的期望回報，可以有效地平衡不同狀態之間的回報。

用 V(s) 輔助 q(s,a) 的計算，沒有 a 會更穩定，所以能幫助 q(s,a) 估計。
 