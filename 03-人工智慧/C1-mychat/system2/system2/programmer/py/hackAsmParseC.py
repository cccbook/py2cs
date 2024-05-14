def parse_c_instruction(instruction):
    comp = "0"  # Placeholder value, replace with actual computation logic
    dest = "000"  # Placeholder value, replace with actual destination logic
    jump = "000"  # Placeholder value, replace with actual jump logic

    if "=" in instruction:
        dest, comp = instruction.split("=")
    if ";" in instruction:
        comp, jump = comp.split(";")
    """
    if "=" in instruction:
        dest, comp = instruction.split("=")
    if ";" in instruction:
        comp, jump = comp.split(";")
    """
    return f"111{comp_code(comp)}{dest_code(dest)}{jump_code(jump)}"

def comp_code(comp):
    comp_table = {
        "0": "0101010",
        "1": "0111111",
        "-1": "0111010",
        "D": "0001100",
        "A": "0110000",
        "!D": "0001101",
        "!A": "0110001",
        "-D": "0001111",
        "-A": "0110011",
        "D+1": "0011111",
        "A+1": "0110111",
        "D-1": "0001110",
        "A-1": "0110010",
        "D+A": "0000010",
        "D-A": "0010011",
        "A-D": "0000111",
        "D&A": "0000000",
        "D|A": "0010101",
    }
    return comp_table[comp]

def dest_code(dest):
    dest_table = {
        "": "000",
        "M": "001",
        "D": "010",
        "MD": "011",
        "A": "100",
        "AM": "101",
        "AD": "110",
        "AMD": "111",
    }
    return dest_table[dest]

def jump_code(jump):
    jump_table = {
        "": "000",
        "JGT": "001",
        "JEQ": "010",
        "JGE": "011",
        "JLT": "100",
        "JNE": "101",
        "JLE": "110",
        "JMP": "111",
    }
    return jump_table[jump]

# 測試案例
if __name__ == "__main__":
    assert parse_c_instruction("D=A") == "1110110000000000"
    assert parse_c_instruction("D;JGT") == "1110000000000001"
    assert parse_c_instruction("M=D+M") == "1111000010001000"
    assert parse_c_instruction("0;JMP") == "1110101010000111"

print("測試通過！")
