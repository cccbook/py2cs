import grammar

G = {
    "S":["NP VP"],
    "NP":["D N"],
    "VP":["V NP"],
    "N":["dog", "cat"],
    "V":["eat", "chase"],
    "D":["a", "the"]
}

grammar.gen(G)
