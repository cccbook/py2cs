# EDA0 -- 用 Python 做的微型 EDA IC 設計工具

我又開始手癢想做個新專案了

1. 包含一個簡化過的的 Verilog 編譯器 (符合 script1 的擴充語法)
2. RTL to Logic Synthesis : 可以將該 script1 語法轉為 RTL
3. Technology Independent Synthesis : 對 RTL 邏輯式進行簡化
4. Technology Dependent Synthesis : 將邏輯式映射到元件庫並最小化
5. Placement & Routing : 決定各元件擺放位置，讓面積與路徑總成本愈小愈好

## 主要參考文獻

* [ECE 5745 -- Topic 12: Synthesis Algorithms (PDF)](https://www.csl.cornell.edu/courses/ece5745/handouts/ece5745-T12-cad-synthesis.pdf)
* [ECE 5745 -- Topic 13: Physical Design Automation Algorithms (PDF)](https://www.csl.cornell.edu/courses/ece5745/handouts/ece5745-T13-cad-physical.pdf)

