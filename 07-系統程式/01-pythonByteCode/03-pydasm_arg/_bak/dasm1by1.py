import dis
import marshal

pyc_file_path = '__pycache__/example.cpython-312.pyc'

# 讀取 .pyc 檔案
with open(pyc_file_path, 'rb') as f:
    f.read(16)  # 跳過魔數和時間戳
    code = marshal.load(f)  # 讀取實際的 bytecode

# 逐個取出指令
for instr in dis.get_instructions(code):
    # print('#', instr)
    start_line = instr.starts_line or ''
    arg = instr.arg or '' 
    print(f'{start_line}\t{instr.offset}\t{instr.opname}\t{arg}')

