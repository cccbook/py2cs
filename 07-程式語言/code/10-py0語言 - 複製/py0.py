import re

def lexer(program):
    # 定義詞法單元的正規表達式
    regex_patterns = [
        (r'\b(?:var)\b', 'VAR'),  # 變數名稱
        (r'\b(?:\d+)\b', 'NUMBER'),  # 數字
        (r'\+', 'PLUS'),  # 加法運算子
        (r'-', 'MINUS'),  # 減法運算子
        (r'\*', 'MULTIPLY'),  # 乘法運算子
        (r'/', 'DIVIDE'),  # 除法運算子
        (r'\s', 'WHITESPACE'),  # 空白字符
    ]

    tokens = []

    # 將正規表達式轉換為 pattern 對象
    patterns = [(re.compile(pattern), token_type) for pattern, token_type in regex_patterns]

    while program:
        found = False
        for pattern, token_type in patterns:
            match = pattern.match(program)
            if match:
                tokens.append((token_type, match.group(0)))
                program = program[match.end():]
                found = True
                break

        if not found:
            raise ValueError(f"Unexpected character: {program[0]}")

    return tokens

# 測試詞彙掃描器
if __name__ == "__main__":
    while True:
        try:
            # 輸入待掃描的程式碼
            program = input("輸入程式碼（輸入 exit 結束）: ")

            if program.lower() == 'exit':
                break

            tokens = lexer(program)
            print("Token List:", tokens)

        except Exception as e:
            print(f'Error: {e}')
