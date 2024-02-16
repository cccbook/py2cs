# https://doc.deno.land/builtin/stable#Deno.readTextFile
with open('hello.txt', 'r') as reader:
    # Read & print the entire file
    print(reader.read())
