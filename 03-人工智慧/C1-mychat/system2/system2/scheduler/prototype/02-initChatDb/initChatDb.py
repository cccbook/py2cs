import sqlite3
import os

def sql_execute(db, sql_cmd):
    cursor = db.cursor()
    cursor.execute(sql_cmd)
    db.commit()

def sql_select(db, sql_cmd):
    cursor = db.cursor()
    cursor.execute(sql_cmd)
    rows = cursor.fetchall()
    return rows
        
DB_FILE = '../chatDb/chat.db'
if os.path.isfile(DB_FILE):
    os.remove(DB_FILE)

talks = [
    ['ccc', '2023/10/01 12:35', '本月 22 日中午我有系務會議要開'],
    ['ccc', '2023/10/02 14:03', '本月 10 日放假一天'],
    ['ccc', '2023/10/07 15:22', '下個月 15 號我要飛泰國，坐華航下午 3:20 的班機去'],
    ['ccc', '2023/10/13 09:22', '明天下午三點接小孩去補習'],
]

db = sqlite3.connect(DB_FILE)
sql_execute(db, f'CREATE TABLE IF NOT EXISTS talks (說話者 TEXT, 時間 TEXT, 內容 TEXT)')
for a in talks:
    sql_execute(db, f'INSERT INTO talks (說話者, 時間, 內容) VALUES("{a[0]}", "{a[1]}", "{a[2]}")')
rows = sql_select(db, f'SELECT * FROM talks')
print('rows=', rows)
db.close()
