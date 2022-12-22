import sqlite3

db = sqlite3.connect('stockPrice.db1')
cursor = db.cursor()
while True:
    print('sql:', end='')
    cmd = input()
    if cmd == 'exit': break
    cursor.execute(cmd)
    rows = cursor.fetchall()
    for row in rows:
        print(row)
db.close()