import re

PATH = './index'

def addIndex(invIndex, word, d):
    # print('add:', word, d)
    bucket = invIndex.get(word)
    if bucket:
        if not d in bucket:
            bucket.append(d)
    else:
        invIndex[word] = [d]

def idxBuild(invIndex, docs):
    for d in range(len(docs)):
        i = 0
        doc = docs[d]
        dlen = len(doc)
        while i<dlen:
            eword = re.compile("[a-zA-Z]+")
            m = eword.match(doc, i)
            word = None
            if m:
                word = m.group(0).lower()
                addIndex(invIndex, word, d)
                i+=len(word)
            else:
                cword = re.compile("[\u4e00-\u9fff]{1,3}") # https://stackoverflow.com/questions/2718196/find-all-chinese-text-in-a-string-using-python-and-regex
                m = cword.match(doc, i)
                if m:
                    word = m.group(0)
                    for wlen in range(1, len(word)+1):
                        addIndex(invIndex, word[0:wlen], d)
                i+=1

def idxSave(invIndex, path=PATH):
    for key, hits in invIndex.items():
        # print(f'{key}:{hits}')
        with open(f'{path}/{key}', mode='ab') as f:
            f.write(bytearray(hits))
            #for hit in hits:
            #    f.write(hit.to_bytes(4, byteorder='big', signed=False))

def idxQuery(word, path=PATH):
    # print(f'idxQuery({word}):')
    with open(f'{path}/{word}', mode='rb') as f:
        hits = list(f.read())
        return hits
