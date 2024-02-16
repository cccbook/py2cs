#include "bit.h"

char mask[8] = {0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01};

int bitcommon(char *a, char *b, int nbit) {
    int i;
    for (i=0; i<nbit/8; i++) {
        if (a[i] != b[i]) break;
    }
    i *= 8;
    while (i<nbit) {
        if (bit(a,i) != bit(b,i)) break;
        i++;
    }
    return i;
}

int bitcmp(char *a, char *b, int nbit) {
    size_t nbyte = nbit/8;
    int r = memcmp(a, b, nbyte);
    if (r!=0) return r;
    int n = nbit%8;
    for (int i=0; i<n; i++) {
        int ai=bit(a,i);
        int bi=bit(b,i);
        if (ai>bi) return 1;
        if (ai<bi) return -1;
    }
    return 0;
}