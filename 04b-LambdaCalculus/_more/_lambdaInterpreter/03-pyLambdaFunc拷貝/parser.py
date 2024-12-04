import re
import json

# Tokenize input
def tokenize(expression):
    token_specification = [
        ("LPAREN", r"\("),      # 左括號
        ("RPAREN", r"\)"),      # 右括號
        ("IDENT", r"[A-Za-z]+"),  # 標識符
        ("SKIP", r"[ \t]+"),    # 跳過空白
    ]
    token_regex = "|".join(f"(?P<{pair[0]}>{pair[1]})" for pair in token_specification)
    for match in re.finditer(token_regex, expression):
        kind = match.lastgroup
        value = match.group()
        if kind == "SKIP":
            continue
        yield kind, value
    yield "EOF", None  # 結束標誌

# 遞迴下降解析器
class Parser:
    def __init__(self, tokens):
        self.tokens = iter(tokens)
        self.current_token = None
        self.next_token()

    def next_token(self):
        self.current_token = next(self.tokens)

    def parse(self):
        return self.expr()

    def expr(self):
        """
        表達式處理，主要是運算的標識符和參數
        """
        if self.current_token[0] == "IDENT":
            identifier = self.current_token[1]
            self.next_token()
            args = []
            while self.current_token[0] in {"IDENT", "LPAREN"}:
                args.append(self.expr())
            return {"type": "CALL", "name": identifier, "args": args}
        elif self.current_token[0] == "LPAREN":
            self.next_token()  # 跳過 '('
            expr = self.expr()
            if self.current_token[0] == "RPAREN":
                self.next_token()  # 跳過 ')'
                return expr
            else:
                raise SyntaxError("缺少右括號")
        else:
            raise SyntaxError(f"無效的語法: {self.current_token}")

# 測試解析器
if __name__ == "__main__":
    code = "IF TRUE (OR FALSE TRUE)"
    tokens = list(tokenize(code))
    parser = Parser(tokens)
    ast = parser.parse()
    print(json.dumps(ast, indent=2))
    # print(ast)
