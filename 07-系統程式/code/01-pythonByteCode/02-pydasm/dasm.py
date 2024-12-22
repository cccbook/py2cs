import dis
import marshal

# 讀取 .pyc 文件
with open('__pycache__/example.cpython-312.pyc', 'rb') as f:
    f.read(16)  # 跳過魔數和時間戳
    code = marshal.load(f)

# 顯示 bytecode
dis.dis(code)
