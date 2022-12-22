# 字串編輯距離

```
PS D:\ccc\ccc109a\se\deno\alg\05-dynamicProgramming\editDistance> deno run editDistance.js
editDistance(ATGATCCG,ATGCAATCCC) = 3
======m=========

[0,1,2,3,4,5,6,7,8,9,10]
[1,0,1,2,3,4,5,6,7,8,9]
[2,1,0,1,2,3,4,5,6,7,8]
[3,2,1,0,1,2,3,4,5,6,7]
[4,3,2,1,1,1,2,3,4,5,6]
[5,4,3,2,2,2,2,2,3,4,5]
[6,5,4,3,2,3,3,3,2,3,4]
[7,6,5,4,3,3,4,4,3,2,3]
[8,7,6,5,4,4,4,5,4,3,3]
================

bx= ATG A TCCG
ax= ATGCAATCCC
```
