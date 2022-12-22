#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "bit.h"

typedef struct patNode {
    int nbit;
    char *key;
    struct patNode *lChild, *rChild;
} patNode;

patNode* patSearch(patNode *t, char *key, int len);
patNode* patInsert(patNode *t, char *key, int len);
void patDump(patNode *t, int level);
void patPrint(patNode *t);