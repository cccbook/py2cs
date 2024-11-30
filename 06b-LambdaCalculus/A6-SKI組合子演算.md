## SKI 組合子演算

SKI 組合子演算是一個簡化的計算系統，源於 Lambda 演算，並且它聲稱所有 Lambda 演算中的運算都可以使用三個基本組合子 S、K 和 I 表達。SKI 演算可以視為一種無類型的 Lambda 演算，這個系統中只使用字母 S、K 和 I 來構建所有的表達式，並且通常會省略圓括號來簡化表達式的書寫。

### 非正式定義

在 SKI 系統中，組合子的定義如下：

- **I**：接受一個參數並返回該參數：
  \[
  I \, x \rightarrow x
  \]

- **K**：接受兩個參數，丟棄第二個參數，返回第一個參數：
  \[
  K \, x \, y \rightarrow x
  \]

- **S**：接受三個參數，將第一個參數應用於第三個參數，然後將結果應用於第二個參數的結果，返回最終結果：
  \[
  S \, x \, y \, z \rightarrow x \, z \, (y \, z)
  \]

### I 可以用 S 和 K 表達

儘管 I 是 SKI 系統中的基本組合子之一，事實上它可以由 S 和 K 表達出來。例如，可以通過下列表達式證明：

\[
S \, K \, K \, x \rightarrow K \, x \, (K \, x) \rightarrow x
\]

這表明 I 並不是這個系統中必須存在的最小化組合子，實際上，只有 S 和 K 就足以構成完備的計算系統。

### 形式定義

在 SKI 系統中，項和推導可以使用以下的形式定義：

- **項**：項的集合 \( T \) 是根據以下規則遞歸定義的：
  - S、K 和 I 是項。
  - 如果 \( \tau_1 \) 和 \( \tau_2 \) 是項，則 \( (\tau_1 \, \tau_2) \) 是項。
  - 任何不由以上規則得到的東西都不是項。

- **推導**：推導是由有效項的有限序列組成，這些項遵循以下規則：
  - 如果推導以 \( \alpha(I \, \beta) \, \iota \) 結束，則推導可以推進為 \( \alpha \, \beta \, \iota \)。
  - 如果推導以 \( \alpha((K \, \beta) \, \gamma) \, \iota \) 結束，則推導可以推進為 \( \alpha \, \beta \, \iota \)。
  - 如果推導以 \( \alpha(((S \, \beta) \, \gamma) \, \delta) \, \iota \) 結束，則推導可以推進為 \( \alpha \, ((\beta \, \delta) \, (\gamma \, \delta)) \, \iota \)。

### 例子

1. **自應用和遞歸**

   \( S \, I \, I \) 是一個自應用的表達式，它接受一個參數並將該參數應用於自身：
   \[
   S \, I \, I \, \alpha \rightarrow I \, \alpha \, (I \, \alpha) \rightarrow \alpha \, \alpha
   \]

   此表達式的有趣之處在於它可以導致遞歸：
   \[
   S \, I \, I \, (S \, I \, I) \rightarrow \alpha \, (\alpha \, \alpha) \rightarrow \alpha \, (\alpha \, (\alpha \, \alpha)) \rightarrow \dots
   \]
   這個特性允許我們表達遞歸的過程。

2. **反轉表達式**

   \( S \, (K \, (S \, I)) \, K \) 是一個反轉表達式，它將後兩個項的順序顛倒：
   \[
   S \, (K \, (S \, I)) \, K \, \alpha \, \beta \rightarrow K \, (S \, I) \, \alpha \, (K \, \alpha) \, \beta \rightarrow S \, I \, (K \, \alpha) \, \beta \rightarrow I \, \beta \, (K \, \alpha \, \beta) \rightarrow I \, \beta \, \alpha
   \]
   這個表達式的最終結果是反轉了 \( \alpha \) 和 \( \beta \) 的順序。

3. **布爾邏輯**

   SKI 組合子演算可以實現布爾邏輯運算，例如 `if-then-else` 結構，其中真（T）和假（F）的定義為：
   - \( T = K \)
   - \( F = K \, I \)

   這樣，布爾邏輯中的各種運算，如 NOT、OR 和 AND，都可以通過 SKI 組合子來表達：
   - **NOT**: \( \text{NOT} = (K \, I) \, K \)
   - **OR**: \( \text{OR} = T \)
   - **AND**: \( \text{AND} = F \)

   這些布爾運算的結果可以在 SKI 表達式中完全實現，證明了 SKI 系統的計算完備性。

### 直覺邏輯

SKI 系統中的組合子 K 和 S 對應於直覺邏輯中的兩個命題邏輯公理：
- **AK**: \( A \to (B \to A) \)
- **AS**: \( (A \to (B \to C)) \to ((A \to B) \to (A \to C)) \)

函數應用對應於肯定前件規則（Modus Ponens）：
- **MP**: 如果 \( A \) 和 \( A \to B \)，則 \( B \)。

這些公理和規則確保了 SKI 組合子演算對於直覺邏輯的蘊涵片段是完備的。

### 總結

SKI 組合子演算是一個強大的計算系統，通過簡化 Lambda 演算中的表達式，使得所有的運算都能以 S、K 和 I 三個基本組合子來實現。這不僅使得它成為一個形式化的邏輯系統，還能夠完整表達布爾邏輯和直覺邏輯。