#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <ctype.h>
#include <sys/stat.h>
#include <unistd.h>
#include <assert.h>
#include "utf8.h"

#define STR_SIZE 256
#define DOC_SIZE 4096
#define DOCS_SIZE DOC_SIZE*10
#define BUF_SIZE 1024
#define HASH_SIZE 4096
#define debug printf

typedef int32_t idx_t;

typedef struct Index {
    bool loaded;
    int idxLen, bufLen;
    idx_t *idx, buf[BUF_SIZE];
} Index;

typedef struct Doc {
    idx_t offset;
    char *doc;
    struct Doc *next;
} Doc;

typedef struct DocCache {
    Doc *docp[HASH_SIZE];
} DocCache;

typedef struct DB {
    char path[STR_SIZE];
    char *meta;
    Index index[HASH_SIZE];
    DocCache dc;
    FILE *dataFile;
    // int dataSize;
} DB;

void dbOpen(DB *db, char *path);
idx_t dbAddDoc(DB *db, char *doc);
void dbFlush(DB *db);
void dbClose(DB *db);
char *dbMatch(DB *db, char *q, char *follow, char *docs, int docsMaxSize);

// https://stackoverflow.com/questions/8666378/detect-windows-or-linux-in-c-c
#ifdef _WIN32
#define mkdir1(path) mkdir(path)
#else
#define mkdir1(path) mkdir(path, 744)
#endif
