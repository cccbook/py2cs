import dis

# 列出所有 opcode 名稱和對應的數字編碼
for code, name in enumerate(dis.opname):
    print(f"{code:>3}: {name}")
