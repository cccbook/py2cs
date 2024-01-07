import re

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
            rcode.append('\t'*level+'<begin>\n')
        if tlevel < level:
            rcode.append('\t'*tlevel+'<end>\n')
        rcode.append(line+'\n')
        level = tlevel
    return ''.join(rcode)

def lex(pcode):
    code = preprocess(pcode)
    # 定義詞法單元的正規表達式
    regex_patterns = [
        (r'\w+', 'ID'),  # 變數名稱
        (r'\d+', 'NUMBER'),  # 數字
        (r'<\w+>', 'TAG'),  # TAG
        (r'[\n\t\s\r]', 'SPACE'),  # 空格
        (r'[\n\t\s\r]', 'SPACE'),  # 空格
        (r'\S', 'CHAR'),  # 單一字元
    ]

    tokens = []

    # 將正規表達式轉換為 pattern 對象
    patterns = [(re.compile(pattern), token_type) for pattern, token_type in regex_patterns]

    while code:
        found = False
        for pattern, token_type in patterns:
            match = pattern.match(code)
            if match:
                if token_type == 'INDENT':
                    tokens.append((token_type, len(match.group(0))-1))
                elif token_type != 'SPACE':
                    tokens.append((token_type, match.group(0)))
                code = code[match.end():]
                found = True
                break

        if not found:
            raise ValueError(f"Unexpected character: {code[0]}")

    return tokens

# 測試詞彙掃描器
if __name__ == "__main__":
    with open("./example/fib.py") as f:
        code = f.read()
        print(code)
    tokens = lex(code)
    print("tokens:", tokens)
