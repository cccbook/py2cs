import grammar

G = {
    "S":["a S b","a S b","a S b","a b"] # 產生 an bn
    # 上面重複三次 a S b 是為了增大該情況的機率為三倍
}

grammar.gen(G)
