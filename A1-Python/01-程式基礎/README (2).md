# 複雜度

* https://www.bigocheatsheet.com/
* https://cooervo.github.io/Algorithms-DataStructures-BigONotation/index.html
* https://en.wikipedia.org/wiki/Best,_worst_and_average_case
* https://en.wikipedia.org/wiki/Search_data_structure#Asymptotic_amortized_worst-case_analysis
* https://en.wikipedia.org/wiki/Sorting_algorithm#Comparison_of_algorithms

## BigO

* [維基百科:大O符號](https://zh.wikipedia.org/wiki/%E5%A4%A7O%E7%AC%A6%E5%8F%B7)

演算法的複雜度通常以 Big O 函數來衡量，Big O 是一個只考慮函數成長等級，但是不考慮常數項的概念。

以下是一些程式的複雜度範例，其中的 n 在未指定的情況下通常是指《輸入資料長度》 (例如陣列長度)。

* O(1) -- [distance.js](bigO/distance.js)
* O(log n) -- [binSearch.js](../06-divideConquer/binSearch.js)
* O(n) -- [lsearch](bigO/lsearch.js)
* O(n log n)-- [binSearch.js](../06-divideConquer/mergeSort/)
* O(n^2) -- [bubbleSort.js](bigO/bubbleSort.js)
* O(n^3) -- [matrixMul.js](bigO/matrixMul.js)
    * 這裡的 n 是指矩陣的高或寬
    * 更精確的寫法應該是 m * n * p
* O(2^n) -- [power2n.js](bigO/power2n.js), [fibonacci.js](../01-tableLookup/fibonacci.js), [sat.js](../16-npcomplete/SAT/sat.js)

## 基本資料結構的複雜度

* https://www.bigocheatsheet.com/

* Array : => LinkedList : 
* HashTable : 
* BinaryTree : => 紅黑樹
* Trie : => PatriciaTrie



