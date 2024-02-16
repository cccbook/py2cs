import grammar

G = {
    "S":["aBC","aSBC","aSBC","aSBC"], # 產生 anBnCn
    "CB":["CZ"], # CB=>CZ=>WZ=>WC=>BC 導致 C 和 B 交換了?
    "CZ":["WZ"],
    "WZ":["WC"],
    "WC":["BC"],
    "aB":["ab"], # 換成小寫
    "bB":["bb"],
    "bC":["bc"],
    "cC":["cc"]
}

grammar.gen(G)


