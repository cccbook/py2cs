# https://doc.deno.land/builtin/stable#Deno.readTextFile
text = """ self is a very
    long string if I had the
    energy to type more and more ..."""

with open('out.txt', 'w') as writer:
    # Read & print the entire file
    writer.write(text)
