import grammar

G = {
    "S":["NP VP"],
    "NP":["D N", "D A N"],
    "VP":["V NP"],
    "A":["black", "white", "little", "big"],
    "N":["dog", "cat"],
    "V":["eat", "chase"],
    "D":["a", "the"]
}

grammar.gen(G)
