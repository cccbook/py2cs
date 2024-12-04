import grammar

G = {
    "S":["a S b","a S b","a S b","a b"] # 產生 anbn
}

grammar.gen(G)
