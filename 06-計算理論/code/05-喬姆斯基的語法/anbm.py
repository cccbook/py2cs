import grammar

G = { # 產生 an bm
    "S":["A"],
    "A":["a A", "a A", "a A", "B"], # 重複三次 a A 是為了增大該情況的機率為三倍
    "B":["b B", "b B", "b B", "b"]  # 重複三次 b B 是為了增大該情況的機率為三倍
}

grammar.gen(G)


