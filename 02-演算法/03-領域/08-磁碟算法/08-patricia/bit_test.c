#include "patricia.h"

int main() {
    char a[] = {0x81,0xF2,0x73,0x00};
    char b[] = {0x81,0xF2,0x72,0x00};
    int nbit = 32;
    for (int i=0; i<nbit; i++) {
        printf("%d", bit(a, i));
    }
    printf("\n");
    printf("bitcmp(a,b)=%d\n", bitcmp(a, b, nbit));
    printf("bitcommon(a,b)=%d\n", bitcommon(a, b, nbit));
}
