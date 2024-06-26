def assemble(inputText, outputText):
    symbol_table = {
        "SP": 0, "LCL": 1, "ARG": 2, "THIS": 3, "THAT": 4,
        "R0": 0, "R1": 1, "R2": 2, "R3": 3, "R4": 4, "R5": 5,
        "R6": 6, "R7": 7, "R8": 8, "R9": 9, "R10": 10, "R11": 11,
        "R12": 12, "R13": 13, "R14": 14, "R15": 15,
        "SCREEN": 16384, "KBD": 24576
    }
    next_variable_address = 16

    def parse_a_instruction(instruction):
        try:
            address = int(instruction[1:])
        except ValueError:
            symbol = instruction[1:]
            if symbol not in symbol_table:
                symbol_table[symbol] = next_variable_address
                address = next_variable_address
                next_variable_address += 1
            else:
                address = symbol_table[symbol]
        return format(address, '016b')

    def parse_c_instruction(instruction):
        comp = "0"  # Placeholder value, replace with actual computation logic
        dest = "000"  # Placeholder value, replace with actual destination logic
        jump = "000"  # Placeholder value, replace with actual jump logic
        return f"111{comp}{dest}{jump}"

    output_lines = []

    for line in inputText.split('\n'):
        line = line.strip()
        if not line or line.startswith('//'):
            continue
        elif line.startswith('@'):
            output_lines.append(parse_a_instruction(line))
        else:
            output_lines.append(parse_c_instruction(line))

    outputText.write('\n'.join(output_lines))

# 測試案例
if __name__ == "__main__":
    input_text = """
    // Sample Assembly Code
    @2
    D=A
    @3
    D=D+A
    @0
    M=D
    """
    with open("output.hack", "w") as output_file:
        assemble(input_text, output_file)
