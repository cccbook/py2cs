import re

tkLevel, tkType, tkWord, tkLine, tkPos = 0, 1, 2, 3, 4

def tabStrLen(s):
    return len(s.replace("\t", "    "))

def lex(code):
    # 定義詞法單元的正規表達式
    regex_patterns = [
        (r'(==)|(!=)|(<<?=?)|(>>?=?)|(\*\*)|(\+=)|(-=)|(\*\*?=)|(//?=)|(%=)', 'KEY'),  # 2 個字元以上的運算
        (r'\w+', 'ID'),  # 變數名稱
        (r'\d+', 'INT'),  # 整數
        (r'\d+.\d*', 'FLOAT'),  # 浮點數
        # (r'<\w+>', 'TAG'),  # TAG
        (r'\n\t*', 'INDENT'),  # 行開頭
        (r'[\r\b\s]+', 'SPACE'),  # 空格
        (r'\S', 'KEY'),  # 單一字元
    ]

    tokens = []

    # 將正規表達式轉換為 pattern 對象
    patterns = [(re.compile(pattern), token_type) for pattern, token_type in regex_patterns]

    line, pos, level = 0, 0, 0
    while code:
        found = False
        for pattern, token_type in patterns:
            match = pattern.match(code)
            if match:
                token = match.group(0)
                if token_type == 'INDENT':
                    hlevel = len(token)-1
                    pos = tabStrLen(token[1:])
                    if len(tokens)>0 and (tokens[-1][tkType] in ['BEGIN', 'END', 'INDENT'])
                        pass
                    elif hlevel > level:
                        tokens.append((level, 'BEGIN', token, level, line))
                    elif hlevel < level:
                        tokens.append((level, 'END', token, level, line))
                    else:
                        tokens.append((level, 'INDENT', token, level, line))
                    # else: tokens.append(('<none>', hlevel, line))
                    line += 1
                    level = hlevel
                elif token_type == 'SPACE':
                    pos += tabStrLen(token)
                elif token_type == 'KEY':
                    tokens.append((level, token_type, token, level, line))
                    pos += len(token)
                else:
                    tokens.append((level, token_type, token, pos, line))
                    pos += tabStrLen(token)
                code = code[match.end():]
                found = True
                break

        if not found:
            raise ValueError(f"Unexpected character: {code[0]}")

    return tokens

def format(code):
    words = []
    tokens = lex(code)
    for token in tokens:
        level, tktype, word, line, pos = token
        tabs = '\t'*level
        if tktype == 'BEGIN':
           words.append('\n'+tabs+'begin\n'+tabs+'\t')
        elif token[tkType] == 'END':
           words.append('\n'+tabs+'end\n'+tabs+'\t')
        elif token[tkType] == 'INDENT':
           words.append('\n'+tabs)
        else:
           words.append(word+' ')
    return ''.join(words)

# 測試詞彙掃描器
if __name__ == "__main__":
    with open("./example/fib.py") as f:
        code = f.read()
        print(code)
    tokens = lex(code)
    print("tokens:", tokens)
    fcode = format(code)
    print(fcode)
