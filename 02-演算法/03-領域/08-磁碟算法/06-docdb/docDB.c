#include "docDB.h"

char *strlower(char *s) {
    for (char *p=s; *p; p++)
        *p = tolower(*p);
    return s;
}

int shash(char *s, int len) {
    int h = 371;
    for (int i=0; i<len; i++) {
        int c = (uint8_t) s[i];
        h = ((h << 5) - h + c)%HASH_SIZE;
    }
    return h;
}

int ihash(int i) {
    return i%HASH_SIZE;
}

int fileSize(FILE *fp) {
    fseek(fp, 0L, SEEK_END);
    return ftell(fp);
}

void indexRead(Index *index, FILE *fp) {
    if (index->loaded) return; // 已經讀過了，直接傳回
    int size = fileSize(fp);
    index->idx = malloc(size);
    index->idxLen = size/sizeof(idx_t);
    fseek(fp, 0L, SEEK_SET);
    fread(index->idx, sizeof(idx_t), index->idxLen, fp);
    // debug("indexRead:size=%d idxLen=%d idx[0]=%d\n", size, index->idxLen, index->idx[0]);
}

void indexFlush(Index *index, FILE *fp) {
    if (index->bufLen == 0) return; // 沒有資料，不用寫入。
    fwrite(index->buf, sizeof(idx_t), index->bufLen, fp);
}

bool indexAdd(Index *index, idx_t offset) {
    if (index->bufLen < BUF_SIZE) {
        index->buf[index->bufLen++] = offset;
        // debug("index->bufLen=%d offset=%d\n", index->bufLen, offset);
        return true;
    } else {
        assert(false);
    }
    return false;
}

bool isDir(char *path) {
    struct stat sb;
    return stat(path, &sb) == 0 && S_ISDIR(sb.st_mode);
}

char *dbReadDoc(DB *db, idx_t offset) {
    fseek(db->dataFile, offset, SEEK_SET);
    char doc[DOC_SIZE];
    fgets(doc, DOC_SIZE-1, db->dataFile);
    return strdup(doc);
}

char *dbGetDoc(DB *db, idx_t offset) {
    int h = ihash(offset);
    Doc *docp = db->dc.docp[h];
    if (docp) {
        for (Doc *p = docp; p; p=p->next) {
            if (p->offset == offset) return p->doc;
        }
    }
    Doc *d1 = malloc(sizeof(Doc));
    d1->doc = dbReadDoc(db, offset);
    d1->next = docp;
    db->dc.docp[h] = d1;
    return d1->doc;
}

idx_t dbWriteDoc(DB *db, char *doc) {
    fseek(db->dataFile, 0, SEEK_END);
    idx_t offset = ftell(db->dataFile);
    debug("dbWriteDoc:offset=%d\n", offset);
    fprintf(db->dataFile, "%s\n", doc);
    // db->dataSize += strlen(doc);
    return offset;
}

Index *dbReadIndex(DB *db, int h) {
    Index *index = &db->index[h];
    if (index->loaded) return index;
    char idxFileName[STR_SIZE+20];
    sprintf(idxFileName, "%s/idx/%d", db->path, h);
    FILE *idxFile = fopen(idxFileName, "r+b");
    indexRead(index, idxFile);
    fclose(idxFile);
    return index;
}

void dbFlushIndex(DB *db, int h) {
    Index *index = &db->index[h];
    if (index->bufLen == 0) return; // 沒有資料，不用寫入。
    char idxFileName[STR_SIZE+20];
    sprintf(idxFileName, "%s/idx/%d", db->path, h);
    FILE *idxFile = fopen(idxFileName, "a+b");
    assert(idxFile);
    indexFlush(index, idxFile);
    fclose(idxFile);
}

void dbIndexWord(DB *db, char *word, int wordLen, idx_t offset) {
    int h = shash(word, wordLen);
    // debug("word=%.*s h=%d\n", wordLen, word, h);
    Index *index = &db->index[h];
    indexAdd(index, offset);
}

void dbIndexDoc(DB *db, char *doc, idx_t offset) {
    assert(strlen(doc)<DOC_SIZE);
    char doc1[DOC_SIZE];
    strcpy(doc1, doc);
    strlower(doc1);
    char *dp = doc1;
    while (*dp) {
        char *p = dp;
        if (isalpha(*dp)) { // english word
            while (isalpha(*p)) p++;
            dbIndexWord(db, dp, p-dp, offset);
            dp = p;
        } else if (isdigit(*dp)) { // number
            while (isdigit(*p)) p++;
            dbIndexWord(db, dp, p-dp, offset);
            dp = p;
        } else if (*dp >= 0) { // other ASCII
            dp++;
        } else { // Non ASCII UTF8 code bytes.
            for (int i=0; i<4; i++) {
                int len = utf8len(p); 
                p += len;
                dbIndexWord(db, dp, p-dp, offset);
            }
            dp += utf8len(dp);
        }
    }
}

idx_t dbAddDoc(DB *db, char *doc) {
    idx_t offset = dbWriteDoc(db, doc);
    dbIndexDoc(db, doc, offset);
    return offset;
}

void dbOpen(DB *db, char *path) {
    memset(db, 0, sizeof(DB));
    strcpy(db->path, path);
    char dataFileName[STR_SIZE];
    sprintf(dataFileName, "%s/jdb.data", path);
    if (isDir(path)) {
        db->dataFile = fopen(dataFileName, "r+");
        db->meta = dbGetDoc(db, 0);
        // db->dataSize = fileSize(db->dataFile);
    } else {
        mkdir1(path); // mkdir(path, 0744);
        char idxPath[STR_SIZE];
        sprintf(idxPath, "%s/idx", path);
        mkdir1(idxPath); // mkdir(idxPath, 0744);
        db->dataFile = fopen(dataFileName, "w+");
        // db->dataSize = 0;
        db->meta = strdup("{db:jdb}");
        dbAddDoc(db, db->meta);
    }
}

void dbClose(DB *db) {
    fclose(db->dataFile);
}

void dbFlush(DB *db) {
    for (int h=0; h<HASH_SIZE; h++) {
        dbFlushIndex(db, h);
    }
}

char *dbMatch(DB *db, char *q, char *follow, char *docs, int docsMaxSize) {
    assert(strlen(q)<STR_SIZE);
    char q1[STR_SIZE];
    strcpy(q1, q);
    strlower(q1);
    // debug("q1=%s\n", q1);
    int h = shash(q1, strlen(q1));
    Index *index = dbReadIndex(db, h);
    // debug("dbMatch: h=%d index->idxLen=%d\n", h, index->idxLen);
    char *dp = docs;
    for (int i=0; i<index->idxLen; i++) {
        // debug("idx[%d]=%d\n", i, index->idx[i]);
        char *doc = dbGetDoc(db, index->idx[i]);
        // debug("doc=%s\n", doc);
        char doc1[DOC_SIZE];
        strcpy(doc1, doc);
        strlower(doc1);
        char *qs = strstr(doc1, q1);
        if (qs && (follow == NULL || strchr(follow, qs[strlen(q1)]))) {
            if (dp-docs+strlen(doc) >= docsMaxSize-1) break;
            sprintf(dp, "%s", doc);
            dp += strlen(dp);
        }
    }
    return docs;
}
