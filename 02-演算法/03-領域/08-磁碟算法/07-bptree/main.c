#include "bptree.h"

// #define N 10000
#define N 50000
char strTable[N*20];
char *stp = strTable;

int main() {
    bptInit();
    for (int k = 0; k<N; k++) {
        // printf("k=%d\n", k);
        // k 的第 1 筆
        sprintf(stp, "%d", 2*k);
        bptInsert(k, stp);
        stp += strlen(stp)+1;
        // k 的第 2 筆
        sprintf(stp, "%d", 2*k+1);
        bptInsert(k, stp);
        stp += strlen(stp)+1;
    }
    // printf("bptFind(3)=%d\n", bptFind(3));
    printf("bptQueryKey(3)=%d\n", bptQueryKey(10));
    printf("bptQueryRange(10,20)=%d\n", bptQueryRange(10,20));
    bptDestroy();
}