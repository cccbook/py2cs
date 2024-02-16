# https://stackoverflow.com/questions/105034/create-guid-uuid-in-javascript
import random

def s4():
  return hex(random.randrange(0, 0x10000))[2:]

def guid():
  return s4() + s4() + '-' + s4() + '-' + s4() + '-' + s4() + '-' + s4() + s4() + s4()

print(guid())