#include <stdio.h>
#include <string.h>
#include "docDB.h"

DB ddb;

int main() {
    utf8init();
    DB *db = &ddb;
    dbOpen(db, "./jdb");
    char docs[DOCS_SIZE];
    dbMatch(db, "UNIX", NULL, docs, DOCS_SIZE);
    printf("docs:\n%s\n", docs);
    dbClose(db);
}
