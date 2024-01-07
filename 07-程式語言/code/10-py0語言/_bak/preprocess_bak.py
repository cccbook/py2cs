def preprocess(code):
    lines = code.split("\n")
    level = 0
    for line in lines:
        if line.strip() == "": continue
        tlevel = len(line) - len(line.lstrip('\t'))
        if tlevel > level:
            print('\t'*level+'{')
        if tlevel < level:
            print('\t'*tlevel+'}')
        print(line)
        level = tlevel


# 測試詞彙掃描器
if __name__ == "__main__":
    with open("./example/fib.py") as f:
        code = f.read()
        print(code)
    preprocess(code)
