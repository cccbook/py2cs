import grammar

G = {
    "S":["NP VP", "NP VP", "NP VP PP"],
    "NP":["D N", "D A N"],
    "VP":["V NP"],
    "PP":["that NP V"],
    "A":["black", "white", "little", "big"],
    "N":["dog", "cat"],
    "V":["eat", "chase"],
    "D":["a", "the"]
}

grammar.gen(G)
