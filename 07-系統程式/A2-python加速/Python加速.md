

* ChatGPT: https://chatgpt.com/c/671c51b2-8500-8012-ad04-c0090eb877e1


* https://numba.pydata.org/
    * 範例 -- https://hub.ovh2.mybinder.org/user/numba-numba-examples-23k3mfu7/notebooks/notebooks/basics.ipynb
* https://pypy.org/
* https://github.com/cslarsen/minijit (小型示範)

## ccc: 有沒有專案將 PVM bytecode 用 JIT 加速執行的？

是的，有一些專案專門針對 Python 的 bytecode 使用即時編譯技術（JIT），以加速 PVM bytecode 的執行。以下是幾個主要的專案：

1. **PyPy**：PyPy 是最著名的 Python JIT 編譯器之一。它包含一個高度優化的 JIT 編譯器，可以將 Python 的 bytecode 即時編譯為機器碼，而不需要逐條解釋執行。PyPy 的 JIT 編譯器基於追蹤（tracing）技術，會在運行期間分析程式碼的執行路徑，並針對熱點路徑生成高效的機器碼。PyPy 通常可以比 CPython 快 4 到 10 倍，特別是在長時間運行或計算密集型的應用中效果顯著。

2. **Numba**：Numba 是專為數值計算和科學計算優化的 JIT 編譯器。Numba 通常與 Python 中的數值運算庫（例如 NumPy）配合使用，能即時將標記過的 Python 函數（特別是數值密集型函數）編譯為機器碼。這主要是針對函數級別的加速，並不針對全局的 Python bytecode，但可以在計算密集型代碼中顯著提升效率。

3. **Pyston**：Pyston 是一個專為 Python 開發的 JIT 編譯器專案，基於 LLVM 編譯技術。Pyston 的目標是提供比 CPython 更高的執行效率，特別針對一些 I/O 操作密集型程式。Pyston v2 引入了 JIT 技術，可以即時將 Python bytecode 編譯為更高效的機器碼，但由於資源和維護原因，它的普及度仍在增長中。

4. **GraalVM**：GraalVM 是由 Oracle 開發的多語言 JIT 編譯器環境，支援多種語言，包括 Python。通過 GraalVM 的 Python 介面，可以將 Python 代碼與其他語言無縫整合，並利用 GraalVM 的高效 JIT 編譯引擎執行 Python 程式碼。GraalVM 的 JIT 技術基於 LLVM，理論上可以提供比 CPython 更高的效能。

5. **LLVM with CPython**：這是一個基於 LLVM 的 JIT 編譯嘗試，利用 LLVM JIT 為 Python bytecode 提供編譯加速。它的基本思路是將 CPython 的 bytecode 解釋器指令映射到 LLVM IR，再由 LLVM 即時編譯成機器碼執行。這類實驗性項目目前尚未成為主流選擇，但是值得關注的技術方向。

在這些專案中，**PyPy** 和 **Numba** 是最為成熟和穩定的 JIT 選擇。
