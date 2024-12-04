import grammar

G = {
    "S":["a B C","a S B C","a S B C","a S B C"], # 產生 anBnCn
    "C B":["C Z"], # CB=>CZ=>WZ=>WC=>BC 導致 C 和 B 交換了?
    "C Z":["W Z"],
    "W Z":["W C"],
    "W C":["B C"],
    "a B":["a b"], # 換成小寫
    "b B":["b b"],
    "b C":["b c"],
    "c C":["c c"]
}

grammar.gen(G)


