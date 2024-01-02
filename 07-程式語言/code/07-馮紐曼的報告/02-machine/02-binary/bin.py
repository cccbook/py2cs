# i = 6277101735386680763835789423176059013767194773182842284081
i = 0xF0F0F0
print('i=', i)
with open('out.bin', 'wb') as file:
    file.write((i).to_bytes(4, byteorder='big', signed=False))

i2 = None
with open('out.bin', 'rb') as file:
    i2 = int.from_bytes(file.read(), byteorder='big')

print('i2=', i2)