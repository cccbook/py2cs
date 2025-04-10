import hashlib
import json

def hash(text):
    m = hashlib.sha256()
    m.update(text.encode('utf-8'))
    return m.hexdigest()

record = {
  'nonce': 0,
  'data': 'john => mary $2.7',
}

jsonText = json.dumps(record)
print('jsonText=', jsonText)
digest = hash(jsonText)

print('digest=', digest)
