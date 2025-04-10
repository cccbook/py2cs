// https://cs.stackexchange.com/questions/63048/what-is-the-difference-between-radix-trees-and-patricia-tries

#include "patricia.h"

patNode *patNodeNew(char *key, int nbit) {
    patNode *node = malloc(sizeof(patNode));
    node->nbit = nbit;
    node->key = key;
    node->lChild = NULL;
    node->rChild = NULL; 
    return node;
}

void patPrint(patNode *t) {
    printf("node:nbit=%d key=%.*s\n", t->nbit, (t->nbit+7)/8, t->key);
}

void patDump(patNode *t, int level) {
    if (!t) return;
    printf("%*c", level, ' '); patPrint(t);
    // printf("%d:", level); patPrint(t);
    if (t->lChild->nbit > t->nbit) // 左子樹若向下指
        patDump(t->lChild, level+1); // 才繼續遞迴印出！
    if (t->rChild->nbit > t->nbit) // 右子樹若沒指向自己
        patDump(t->rChild, level+1); // 才繼續遞迴印出！
}

patNode* patSearch(patNode *t, char *key, int nbit) {
    if (!t) return NULL;
    patNode *next = t->lChild;
    patNode *current = t;
    while (next->nbit > current->nbit) { // 當鏈結向上指的時候，就到樹葉節點了。
        current = next;
        next = (bit(key, next->nbit))
                        ? next->rChild
                        : next->lChild;
    }
    return next; // 傳回該樹葉節點。
}

patNode* patInsert(patNode *root, char *key, int nbit) {
    printf("patInsert:nbit=%d key=%.*s\n", nbit, (nbit+7)/8, key);
    patNode *node;
    if (!root) { // 只有一個節點時，左右子樹都指向自己！
        node = patNodeNew(key, 0);
        node->lChild = node;
        node->rChild = node;
        return node;
    }
    patNode *last = patSearch(root, key, nbit);
    int dbit = bitcommon(key, last->key, nbit);
    if (last->key == key || dbit == nbit) {
        printf("Key already Present\n");
        return last;
    }
    patNode *current = root->lChild;
    patNode *parent = root;
    // 再搜尋一次，找出 parent, current
    while (current->nbit > parent->nbit //尚未到樹葉 (向上指代表樹葉)
        && current->nbit < dbit) // 也未到差異位元
    { // 就繼續往下找
        parent = current;
        current = (bit(key, current->nbit))
                        ? current->rChild
                        : current->lChild;
    }
    node = patNodeNew(key, dbit);
    node->lChild = bit(key, dbit) ? current : node;
    node->rChild = bit(key, dbit) ? node : current;
    if (current == parent->lChild) {
        parent->lChild = node;
    } else {
        parent->rChild = node;
    }
    return node;
}
