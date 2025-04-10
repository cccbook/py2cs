## ccc: Haskell Brooks Curry 有何貢獻

Haskell Brooks Curry 是一位美國數學家和邏輯學家，他的研究對數理邏輯和計算理論有著深遠的影響，特別是在函數式編程（Functional Programming）領域。以下是他主要的貢獻：

### 1. **Curry-Howard 同構**
Curry 最著名的貢獻之一是與計算機科學和邏輯學的深刻聯繫，稱為 **Curry-Howard 同構**（Curry-Howard Isomorphism）。這是一種在邏輯和計算機科學之間的對應，將邏輯推理（證明）和計算（程式）聯繫起來。簡單來說，它表明了邏輯定理與程式類型系統之間的對應關係：邏輯命題對應於類型，而證明則對應於程式。這一思想是現代函數式編程語言中類型系統的理論基礎。

### 2. **Combinatory Logic（組合邏輯）**
Curry 與同事 Robert Feys 一同發展了 **組合邏輯**（Combinatory Logic），這是一種基於純函數轉換的數學形式系統。組合邏輯的主要目的是通過去除變量來簡化 Lambda 演算的表述。它使用了一組稱為「組合子」（combinators）的基本函數，通過這些函數的組合來表達其他函數。組合邏輯在計算理論中具有重要地位，並影響了後來的函數式編程語言的設計。

### 3. **Curry 的函數式編程思想**
Haskell Curry 強烈影響了函數式編程的發展，特別是他對函數本質的研究。他將函數視為一等公民的觀點，直接影響了現代 FP 語言的設計。例如，在 FP 中，函數不僅可以作為參數傳遞，還可以作為返回值，這種高階函數的概念在 Curry 的工作中得到了初步的理論支持。

### 4. **Lambda 演算與函數推理的貢獻**
Curry 的研究還擴展了對 Lambda 演算的理解。他試圖將數學中的邏輯推理與函數計算聯繫起來，這為後來計算機科學中的程式語言理論奠定了基礎。Lambda 演算成為了函數式編程中至關重要的概念，並且至今仍是許多 FP 語言（例如 Haskell）的基礎。

### 5. **影響 Haskell 語言的命名**
現代函數式編程語言 **Haskell** 便是以 Haskell Curry 的名字命名的，以表彰他在函數式編程領域的奠基性貢獻。Haskell 語言在設計上深受 Curry 和 Lambda 演算的影響，尤其在純函數式編程和類型系統的應用上。

### 總結
Haskell Brooks Curry 在邏輯學、數學和計算理論方面的貢獻深遠，特別是他對函數和邏輯推理的研究，成為了現代函數式編程語言的理論基礎。他的工作促成了組合邏輯和 Curry-Howard 同構的發展，並對 Lambda 演算的理解進行了擴展，這些都對計算理論及程式語言設計產生了深刻影響。

## ccc: 所以 Church 的年代沒有 Y-Combinator 是嗎？

其實，**Y-Combinator** 的概念在阿隆佐·邱奇（Alonzo Church）的時代已經存在。Church 所創立的 Lambda 演算（Lambda Calculus）提供了 Y-Combinator 的理論基礎。Y-Combinator 是 Lambda 演算中的一種固定點組合子（Fixed-point Combinator），可以用來構建遞歸函數。Y-Combinator 的發現和發展出現於 1930 年代，即邱奇提出 Lambda 演算的時期。

不過，雖然 **Y-Combinator** 的理論最早出現在邱奇的 Lambda 演算中，但這個特定的組合子概念是由 **Haskell Curry** 等人進一步發展和闡明的。因此，Y-Combinator 是在邱奇的理論框架內建立起來的，只是具體的應用和名稱是在後來的計算理論研究中才逐漸被明確和普及。

總結來說，**Y-Combinator** 的理論在 Church 的時代就已經出現，但其具體的名稱和更系統的研究是在稍後的發展中由其他學者，如 Haskell Curry 等人，推廣和應用的。

