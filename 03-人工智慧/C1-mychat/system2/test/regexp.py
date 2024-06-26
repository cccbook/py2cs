import re

value = "123 456 7890"

# Loop over all matches found.
for m in re.finditer("\d+", value):
    print(m.group(0))
    print("start index:", m.start())