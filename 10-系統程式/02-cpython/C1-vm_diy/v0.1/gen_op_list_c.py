import dis

# 列出所有操作碼名稱
op_names = dis.opname

# 印出所有操作碼名稱
print("""
char *op_names[]={
""")
for i in range(len(op_names)):
    print(f'"{op_names[i]}", // {i}')
print('};')