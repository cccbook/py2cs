import dis
import marshal
import sys
# 讀取 .pyc 文件
with open(sys.argv[1], 'rb') as f:
    f.read(16)  # 跳過魔數和時間戳
    code = marshal.load(f)

# 顯示 bytecode
dis.dis(code)
