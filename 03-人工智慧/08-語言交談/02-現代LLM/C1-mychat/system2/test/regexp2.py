import re

text = "3+5 = <python>eval('3+5')</python> ， 5 乘以 5 是 <python>eval('5*5')</python> ，對嗎？"

def replace_code(text):
    parts = []
    last_idx = 0
    for m in re.finditer("<python>(.*?)</python>", text):
        parts.append(text[last_idx:m.start()])
        # print(m.group(0))
        # print("start index:", m.start())
        code = m.group(1)
        # print(code)
        # print(type(code))
        parts.append(str(eval(code)))
        last_idx = m.end()
        print('last_idx=', last_idx)
    parts.append(text[last_idx:])
    print('parts=', parts)
    return ''.join(parts)

print(replace_code(text))

