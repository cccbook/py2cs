from jsondb import JsonDB
import time

lines = []
with open('lines.txt', encoding='utf-8') as f:
    for line in f:
        line1 = line.replace('\n', ' ').strip()
        if len(line1)>0:
            lines.append(line1)

# N = 10000
N = len(lines)*10
jdb = JsonDB()
jdb.open()
n = len(lines)
for i in range(N):
    record = { 'id':i, 'time':time.time(), 'text':lines[i%n] }
    jdb.addObj(record)
jdb.flush()
jdb.close()