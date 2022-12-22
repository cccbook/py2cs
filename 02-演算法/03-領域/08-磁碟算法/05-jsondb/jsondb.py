import json
import re
import os

INT_SIZE = 4
HASH_SIZE = 4096
LOWER = 'abcdefghijklmnopqrstuvwxyz'
DIGITS = '0123456789'
LETTERS = LOWER+LOWER.upper()
LETTER_DIGIT = DIGITS+LETTERS

def hash1(s): # python 的 hash 每個 session 都不一樣 (為了防駭客攻擊)，所以不能用
    s = s.lower()
    h = 0
    for i in range(len(s)):
        h = int(((h << 5) - h) + ord(s[i])) # hash = hash*31 + chr = (hash*32-hash) + c
    return h%HASH_SIZE

def readInt(file):
    ibytes = file.read(INT_SIZE)
    if len(ibytes) == 0: return None
    return int.from_bytes(ibytes)

def writeInt(file, n):
    file.write(n.to_bytes(INT_SIZE))

def fileSize(file):
    file.seek(0, os.SEEK_END)
    return file.tell()

class QueryResults:
    def __init__(self, items):
        self.items = items

    def filter(self, f):
        fitems = filter(f, self.items)
        return QueryResults(fitems)

    def where(self, f):
        return self.filter(lambda x:f(x['obj']))

    def sort(self, byKey, order="INC"):
        return QueryResults(sorted(self.items, key=lambda item:item['obj'][byKey], reverse=order=="DESC"))
    
    def match(self, q):
        return self.filter(lambda x:x['doc'].find(q)>=0)

    def toList(self):
        return list(map(lambda x:x['obj'], self.items))

    def __str__(self):
        return '\n'.join(map(lambda x:(x['doc']), self.items))

class Index:
    def __init__(self):
        self.readed = False
        self.addCount = 0
        self.idx = []

    def read(self, idxFile):
        if self.readed: return # 已經讀過了，直接傳回
        while True:
            offset = readInt(idxFile)
            if offset is None: break
            self.idx.append(offset)
        self.readed = True

    def flush(self, idxFile):
        if self.addCount == 0: return # 沒有資料，不用寫入。
        iLen = len(self.idx)
        for i in range(iLen-self.addCount, iLen):
            writeInt(idxFile, self.idx[i])

    def add(self, offset):
        if len(self.idx)==0 or offset>self.idx[-1]:
            self.idx.append(offset)
            self.addCount += 1
            return True
        return False

class JsonDB:
    def __init__(self):
        pass

    def open(self, path='./jdb'):
        self.index = [Index() for h in range(HASH_SIZE)]
        self.doc = {}
        self.path = path
        dataFileName = f'{self.path}/jdb.data'

        if os.path.isdir(path):
            self.dataFile = open(dataFileName, 'r+', encoding='utf8')
            doc = self.getDoc(0)
            self.meta = json.loads(doc)
            self.dataSize = os.path.getsize(dataFileName)
        else:
            os.mkdir(path)
            os.mkdir(path+'/idx')
            self.dataFile = open(dataFileName, "w+", encoding='utf8')
            self.dataSize = 0
            self.meta = {'db':'jsondb', 'hashSize':HASH_SIZE, 'intSize': INT_SIZE}
            self.addObj(self.meta)

    def close(self):
        self.dataFile.close()

    def flush(self):
        for h in range(HASH_SIZE):
            self.flushIndex(h)

    def readIndex(self, h):
        index = self.index[h]
        if index.readed: return index
        idxFile = open(f'{self.path}/idx/{h}', 'r+b')
        index.read(idxFile)
        idxFile.close()
        return index

    def flushIndex(self, h):
        index = self.index[h]
        if index.addCount == 0: return # 沒有資料，不用寫入。
        idxFile = open(f'{self.path}/idx/{h}', 'a+b')
        index.flush(idxFile)
        idxFile.close()

    def getDoc(self, offset):
        if self.doc.get(offset) is not None: # 已經讀進來到 self.doc 過了，直接傳回就好。
            return self.doc.get(offset)
        return self.readDoc(offset)

    def readDoc(self, offset):
        self.dataFile.seek(offset)
        doc = self.dataFile.readline()
        return doc.replace('\n', '')

    def writeDoc(self, doc):
        self.dataFile.seek(self.dataSize)
        record = doc.replace("\n", "\\n")+'\n'
        self.dataFile.write(record)
        offset = self.dataSize
        self.dataSize += len(str.encode(record))
        return offset

    def addObj(self, obj):
        doc = json.dumps(obj, separators=(',', ':'), ensure_ascii=False) # json.dumps(obj, ensure_ascii=False)
        offset = self.writeDoc(doc)
        self.indexDoc(doc, offset)

    def indexWord(self, word, offset):
        h = hash1(word)
        index = self.index[h]
        index.add(offset)

    def indexDoc(self, doc, offset):
        dlen = len(doc)
        i = 0
        while i<dlen:
            if doc[i]=='"':
                r = re.compile("(\"[a-z]+\":((\d+)|(\"[a-z0-9]+\")))", re.IGNORECASE)
                m = r.match(doc, i)
                if m:
                    field = m.group(0).lower()
                    self.indexWord(field, offset)
            elif doc[i] in LETTER_DIGIT:
                r = re.compile("([a-z]+)|([0-9]+)", re.IGNORECASE)
                m = r.match(doc, i)
                if m:
                    word = m.group(0).lower()
                    self.indexWord(word, offset)
                    i+=len(word)-1
            elif doc[i]>='\u4e00' and doc[i]<='\u9fff':
                r = re.compile("[\u4e00-\u9fff]{2,4}") # https://stackoverflow.com/questions/2718196/find-all-chinese-text-in-a-string-using-python-and-regex
                m = r.match(doc, i)
                if m:
                    word = m.group(0)
                    for wlen in range(1, len(word)+1):
                        self.indexWord(word[0:wlen], offset)
            i+=1

    def match(self, q, follow=None):
        q = q.lower()
        h = hash1(q)
        index = self.readIndex(h)
        r = []
        for offset in index.idx:
            doc = self.getDoc(offset)
            qi = doc.lower().find(q)
            if qi>=0 and (follow is None or doc[qi+len(q)] in follow): # letters, digits or chinese
                r.append({'obj':json.loads(doc), 'doc':doc})
        return QueryResults(r)

    def select(self, key, value):
        if isinstance(value, str): value = f'"{value}"'
        return self.match(f'"{key}":{value}', follow=",}")
