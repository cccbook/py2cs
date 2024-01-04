import grammar

G = { # 產生 an bn cn
    "S":["a B C","a S B C","a S B C","a S B C"],
    # 上面重複三次 a S B C 是為了增大該情況的機率為三倍
    "C B":["C Z"], # CB=>CZ=>WZ=>WC=>BC 導致 C 和 B 交換
    "C Z":["W Z"],
    "W Z":["W C"],
    "W C":["B C"],
    "a B":["a b"], # 換成小寫
    "b B":["b b"],
    "b C":["b c"],
    "c C":["c c"]
}

grammar.gen(G)
