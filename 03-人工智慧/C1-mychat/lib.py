import re
from datetime import datetime, timedelta
from calendar import day_name
import db

def run(code):
    try:
        if code.find('\n')!=-1: raise
        exec(f'_rrr999={code}')
        return str(eval('_rrr999'))
    except:
        return f"<run>{code}</run>"

def replace_code(text):
    parts = []
    last_idx = 0

    for m in re.finditer("<python>(.*?)</python>", text):
        parts.append(text[last_idx:m.start()])
        code = m.group(1)
        parts.append(run(code))
        last_idx = m.end()

    parts.append(text[last_idx:])
    return ''.join(parts)

# sqlite datetime
# https://tableplus.com/blog/2018/07/sqlite-how-to-use-datetime-value.html
def now():
    t = datetime.now()
    # return f"{t.strftime('%m/%d/%Y, %H:%M:%S')} {calendar.day_name[t.weekday()]}"
    return f"{t.strftime('%Y/%m/%d, %H:%M:%S')} {day_name[t.weekday()]}"

# =============== plugin ======================
def system2(question):
    return f"<system2>{question}</system2>"

def calendar(datetime, job):
    db.calendar_add(datetime, job)
    return f"<calendar>{datetime}:{job}</calendar>"

def memory(datetime, event):
    db.memory_add(datetime, event)
    print(f"<memory>{datetime}:{event}</memory>")
