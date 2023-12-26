import grammar

G = {
    "S":["A"],
    "A":["aA", "aA", "aA", "B"],
    "B":["bB", "bB", "bB", "b"]
}

grammar.gen(G)


