import dis

# 列出所有操作碼名稱
op_names = dis.opname

# 印出所有操作碼名稱
print("""
extern char *op_names[];
""")
for i in range(len(op_names)):
    if (op_names[i][0] != '<'):
        print(f'#define {op_names[i]} {i}')
