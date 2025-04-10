https://chatgpt.com/c/67e3781a-5e84-8012-bd82-e8c66054b929

## ccc:  precision 和 recall 的定義

Precision（精確率）和 Recall（召回率）是評估分類模型性能的兩個重要指標，特別適用於不平衡資料集的情境。

### Precision（精確率）
精確率衡量模型預測為正類（Positive）的結果中，有多少是真正的正類。公式如下：
\[
\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}
\]
其中：
- \( \text{TP} \)（True Positive）：被正確分類為正類的樣本數。
- \( \text{FP} \)（False Positive）：被錯誤分類為正類的負類樣本數。

高精確率表示模型在判定為正類時，錯誤較少。

### Recall（召回率）
召回率衡量實際的正類樣本中，有多少被模型正確預測為正類。公式如下：
\[
\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}
\]
其中：
- \( \text{TP} \)（True Positive）：被正確分類為正類的樣本數。
- \( \text{FN} \)（False Negative）：被錯誤分類為負類的正類樣本數。

高召回率表示模型能夠識別大部分的正類樣本，但可能會帶來較多的假陽性。

### Precision 與 Recall 的取捨
- 提高 **Precision** 可能會降低 **Recall**（因為模型變得更嚴格，只在高度確信時才預測為正類）。
- 提高 **Recall** 可能會降低 **Precision**（因為模型更傾向於預測為正類，導致更多錯誤的正類預測）。

這種取捨通常可以透過 **F1-score** 來綜合評估：
\[
F1 = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
\]
F1-score 是 Precision 和 Recall 的調和平均，當兩者都較高時，F1-score 也會較高。

如果是 **不平衡數據集**，如詐欺偵測、醫療診斷等，Precision 和 Recall 比單純的準確率（Accuracy）更重要。