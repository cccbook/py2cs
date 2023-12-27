import grammar

G = {
    "S":["A"],
    "A":["a A", "a A", "a A", "B"],
    "B":["b B", "b B", "b B", "b"]
}

grammar.gen(G)


