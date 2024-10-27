import dis
import marshal
import sys

# 讀取 .pyc 文件
with open(sys.argv[1], 'rb') as f:
    f.read(16)  # 跳過魔數和時間戳
    code = marshal.load(f)

    constants = code.co_consts
    
    # 獲取名稱區域
    names = code.co_names
    
    print("Constants:")
    for i, const in enumerate(constants):
        print(f"{i}  {const} {type(const)}")
    
    print("\nNames:")
    for i, name in enumerate(names):
        print(f"{i}  {name}")

# 顯示 bytecode
dis.dis(code)
