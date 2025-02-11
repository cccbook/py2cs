import marshal
import struct
import dis

# 定義 pyc 檔案中指令的 offset
HEADER_SIZE = 16  # Python 3.12 的表頭大小為 16 bytes

def read_pyc(file_path):
    with open(file_path, "rb") as f:
        # 讀取表頭資訊
        header = f.read(HEADER_SIZE)
        
        # 檢查表頭是否符合期望
        magic_number, python_version = struct.unpack('H2xH2x', header[:8])
        print(f"Magic Number: {magic_number}, Python Version: {python_version}")
        
        # 跳過檔案大小與其他表頭資料，準備讀取後續內容
        code_object = marshal.load(f)  # 反序列化字節碼主程式

    return code_object

def disassemble_code_object(code_obj):
    print(f"\nDisassembling code object: {code_obj.co_name}")
    print(f"Filename: {code_obj.co_filename}")
    print(f"First Line Number: {code_obj.co_firstlineno}")
    print(f"Constants: {code_obj.co_consts}")
    print(f"Names: {code_obj.co_names}")
    print(f"Variable Names: {code_obj.co_varnames}")
    print(f"Free Vars: {code_obj.co_freevars}")
    print(f"Cell Vars: {code_obj.co_cellvars}")
    print(f"Line Number Table (co_lines): {code_obj.co_lines}")
    
    # 反組譯每一個字節碼指令
    for instr in dis.get_instructions(code_obj):
        print(f"{instr.offset:4} {instr.opname:20} {instr.argrepr}")

def main(file_path):
    # 讀取並解析 pyc 檔案
    code_object = read_pyc(file_path)
    # 反組譯字節碼
    disassemble_code_object(code_object)
    
    # 反組譯每一個函數或內嵌的 code object
    for const in code_object.co_consts:
        if isinstance(const, type(code_object)):
            disassemble_code_object(const)

# 執行主函數，提供 .pyc 檔案路徑
if __name__ == "__main__":
    main("__pycache__/example.cpython-312.pyc")
