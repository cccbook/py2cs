import sqlite3

CONNECT = None
CURSOR = None

def open(dbName):
    global CONNECT, CURSOR
    CONNECT = sqlite3.connect(dbName)
    CURSOR = CONNECT.cursor()
    CURSOR.execute("CREATE TABLE IF NOT EXISTS memory (id INTEGER PRIMARY KEY AUTOINCREMENT, datetime TEXT, event TEXT)")
    
    # CURSOR.execute("CREATE TABLE IF NOT EXISTS memory (datetime TEXT, event TEXT)")
    # CURSOR.execute("CREATE TABLE IF NOT EXISTS calendar (id INTEGER PRIMARY KEY AUTOINCREMENT, datetime TEXT, job TEXT, pid INTEGER)")
    CURSOR.execute("CREATE TABLE IF NOT EXISTS calendar (id INTEGER PRIMARY KEY AUTOINCREMENT, datetime TEXT, job TEXT)")
    CONNECT.commit()

def execute(command):
    CURSOR.execute(command)
    CONNECT.commit()

def query(query):
    res = CURSOR.execute(query)
    # res.fetchall()
    return res

def memory_add(datetime, event):
    CURSOR.execute(f"INSERT INTO memory (datetime, event) VALUES ('{datetime}', '{event}')")

def calendar_add(datetime, job):
    CURSOR.execute(f"INSERT INTO calendar (datetime, job) VALUES ('{datetime}', '{job}')")

def dump():
    print('================= memory =====================')
    for row in CURSOR.execute("SELECT id, datetime, event FROM memory ORDER BY datetime"):
        print(row)
    print('================= calendar =====================')
    for row in CURSOR.execute("SELECT id, datetime, job FROM calendar ORDER BY datetime"):
        print(row)

def close():
    CONNECT.commit()
    CONNECT.close()
