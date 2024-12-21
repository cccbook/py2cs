# Context Sensitive Grammar

## 範例 anbncn

## 版本1 -- 會死掉

```
$ python anbncn_v1.py
aSBC
aaSBCBC
aaaSBCBCBC
aaaaBCBCBCBC
aaaabCBCBCBC
aaaabcBCBCBC
aaaabcBCZCBC
aaaabcBCZCZC
aaaabcBWZCZC
aaaabcBWZWZC
aaaabcBWCWZC
aaaabcBWCWCC
aaaabcBBCWCC
aaaabcBBCBCC
aaaabcBBCZCC
aaaabcBBWZCC
aaaabcBBWCCC
aaaabcBBBCCC
```

然後 cB 就換不回來了。

所以不能隨便換，前面的規則可用時，不能套用後面的規則 (前面優先法則)

## 版本 2 -- 採用前面優先法則

```
$ python anbncn_v2.py
aBC
abC
abc
$ python anbncn_v2.py
aSBC
aaSBCBC
aaaSBCBCBC
aaaaSBCBCBCBC
aaaaaBCBCBCBCBC
aaaaaBCZCBCBCBC
aaaaaBCZCZCBCBC
aaaaaBCZCZCZCBC
aaaaaBCZCZCZCZC
aaaaaBWZCZCZCZC
aaaaaBWZWZCZCZC
aaaaaBWZWZWZCZC
aaaaaBWZWZWZWZC
aaaaaBWCWZWZWZC
aaaaaBWCWCWZWZC
aaaaaBWCWCWCWZC
aaaaaBWCWCWCWCC
aaaaaBBCWCWCWCC
aaaaaBBCBCWCWCC
aaaaaBBCZCWCWCC
aaaaaBBWZCWCWCC
aaaaaBBWCCWCWCC
aaaaaBBBCCWCWCC
aaaaaBBBCCBCWCC
aaaaaBBBCCZCWCC
aaaaaBBBCWZCWCC
aaaaaBBBCWCCWCC
aaaaaBBBCBCCWCC
aaaaaBBBCZCCWCC
aaaaaBBBWZCCWCC
aaaaaBBBWCCCWCC
aaaaaBBBBCCCWCC
aaaaaBBBBCCCBCC
aaaaaBBBBCCCZCC
aaaaaBBBBCCWZCC
aaaaaBBBBCCWCCC
aaaaaBBBBCCBCCC
aaaaaBBBBCCZCCC
aaaaaBBBBCWZCCC
aaaaaBBBBCWCCCC
aaaaaBBBBCBCCCC
aaaaaBBBBCZCCCC
aaaaaBBBBWZCCCC
aaaaaBBBBWCCCCC
aaaaaBBBBBCCCCC
aaaaabBBBBCCCCC
aaaaabbBBBCCCCC
aaaaabbbBBCCCCC
aaaaabbbbBCCCCC
aaaaabbbbbCCCCC
aaaaabbbbbcCCCC
aaaaabbbbbccCCC
aaaaabbbbbcccCC
aaaaabbbbbccccC
aaaaabbbbbccccc
```