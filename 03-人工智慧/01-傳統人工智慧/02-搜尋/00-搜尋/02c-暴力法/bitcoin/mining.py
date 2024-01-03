import hashlib
import json
import random

def hash(text):
    m = hashlib.sha256()
    m.update(text.encode('utf-8'))
    return m.hexdigest()

record = {
  'nonce': 0,
  'data': 'john => mary $2.7',
}

def mining(record) :
    for i in range(1000000000000):
        # record['nonce'] = i
        record['nonce'] = random.randint(0,1000000000000)
        h = hash(json.dumps(record))
        if h.startswith('00000'):
            return {'record': record, 'hash': h}

print(mining(record))
