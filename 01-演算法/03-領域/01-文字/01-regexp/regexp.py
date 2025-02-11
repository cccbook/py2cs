# https://docs.python.org/3/library/re.html
import re

s = "24.1632"
m = re.match(r"(\d+)\.(\d+)", s)
print('s=', s, 'groups=', m.groups())

s = "Malcolm Reynolds"
m = re.match(r"(?P<first_name>\w+) (?P<last_name>\w+)", s)
print('s=', s, 'groups=', m.groups())