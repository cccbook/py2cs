import grammar

G = {
    "S": ["E"],
    "E": ["T", "T", "T", "T", "T + E", "T - E", "T * E", "T / E"],
    "T": ["N", "N", "( E )"],
    "N": ["x", "y", "0", "1", "2"]
}

grammar.gen(G)
