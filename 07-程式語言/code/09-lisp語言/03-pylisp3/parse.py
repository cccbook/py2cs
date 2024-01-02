def parse_lisp(code):
    tokens = tokenize(code)
    return parse_tokens(tokens)

def tokenize(code):
    # 將程式碼拆分成標記
    return code.replace('(', ' ( ').replace(')', ' ) ').split()

def parse_tokens(tokens):
    if len(tokens) == 0:
        raise SyntaxError("Unexpected end of input")

    token = tokens.pop(0)

    if token == '(': # 這個設計的思維很妙，遞迴直到 ( 才開始 append ...
        # 如果是左括號，則建立一個新的 list
        parsed_list = []
        while tokens[0] != ')':
            parsed_list.append(parse_tokens(tokens))
        tokens.pop(0)  # 移除右括號
        return parsed_list
    elif token == ')':
        raise SyntaxError("Unexpected ')'")  # 應該不會達到這裡
    else:
        # 其他情況，返回 atom（數字或符號）
        return atom(token)

def atom(token):
    try:
        # 嘗試解析成數字
        return int(token)
    except ValueError:
        # 不是數字，視為符號
        return token

# 測試程式碼轉換
if __name__ == "__main__":
    lisp_code = "(+ 2 (* 3 4))"
    parsed_list = parse_lisp(lisp_code)
    print(parsed_list)
