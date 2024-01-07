def preprocess(code):
    rcode = []
    lines = code.split("\n")
    level = 0
    for line in lines:
        if line.strip() == "":
            rcode.append('\n')
            continue
        tlevel = len(line) - len(line.lstrip('\t'))
        if tlevel > level:
            rcode.append('\t'*level+'<BLOCK>\n')
        if tlevel < level:
            rcode.append('\t'*tlevel+'</BLOCK>\n')
        rcode.append(line+'\n')
        level = tlevel
    return ''.join(rcode)

# 測試詞彙掃描器
if __name__ == "__main__":
    with open("./example/fib.py") as f:
        code = f.read()
        print(code)
    rcode = preprocess(code)
    print(rcode)
