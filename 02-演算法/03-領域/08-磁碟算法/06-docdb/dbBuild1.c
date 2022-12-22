#include <stdio.h>
#include <string.h>
#include "docDB.h"

DB ddb;

char *lines[1000];
int lineTop = 0;

int main() {
    utf8init();
    DB *db = &ddb;
    dbOpen(db, "./jdb");
    FILE *fp = fopen("line.txt", "r+");
    while (!feof(fp)) {
        char line[DOC_SIZE];
        fgets(line, DOC_SIZE, fp);
        line[strcspn(line, "\r\n")] = 0; // remove new line
        if (strlen(line)>0) {
            lines[lineTop++] = strdup(line);
            debug("%s\n", line);
        }
    }
    fclose(fp);
    int n = lineTop * 2;
    for (int i=0; i<n; i++) {
        char doc[DOC_SIZE];
        sprintf(doc, "{id:%d text:\"%s\"}", i, lines[i%lineTop]);
        debug("%s\n", doc);
        dbAddDoc(db, doc);
    }
    dbFlush(db);
    dbClose(db);
}
