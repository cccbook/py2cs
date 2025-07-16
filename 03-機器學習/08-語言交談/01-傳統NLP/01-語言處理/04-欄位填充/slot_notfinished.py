wordMap = {
    "dog":"animal",
    "cat":"animal",
    "people":"animal",
    "john":"man",
    "mary":"woman",
    "jack":"people",
    "man":"people",
    "woman":"people",
    "cookie":"food",
    "beef":"food",
    "eat":"V",
    "chase":"V",
    "emotion":"V",
    "love":"emotion",
    "hate":"emotion",
    "marry":"V",
    "animal":"N",
}

grammars = [
"animal eat food",
"people emotion people",
"woman born child",
"N V N",
]

scripts = [
"people love people; people marry people; woman born people",
]

def match(word, tag):
    if word == None: return False
    if word == tag: return True
    return match(wordMap[word], tag)

def fill(line):
    words = line.split(" ")
    for grammar in grammars:
        slots = grammar.split(" ")
        for word in words:
            if match(word, slots):

def script(text):
    lines = text.split(";")
    for line in lines:
        fill()
