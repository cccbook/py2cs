import os

def match(text, keys):
    for key in keys:
        if text.find(key)>=0:
            return True
    return False

def agent(said):
    cmd = ""
    if match(said, ['youtube']):
        cmd = 'start http://youtube.com'
    if match(said, ['facebook', '臉書']):
        cmd = 'start http://facebook.com'
    print('cmd=', cmd)
    if cmd != "": os.system(cmd)
    return cmd

while True:
    said = input('> ')
    response = agent(said)
    if response is None:
        break
    print(response)
print('bye!')