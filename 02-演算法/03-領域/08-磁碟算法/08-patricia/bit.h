#pragma once
#include <string.h>
#include <stddef.h>

extern char mask[8];
#define min(x, y) (((x) < (y)) ? (x) : (y))
#define max(x, y) (((x) > (y)) ? (x) : (y))
#define bit(s, i) ((s[i/8]&mask[i%8])!=0)

int bitcmp(char *a, char *b, int nbit);
int bitcommon(char *a, char *b, int nbit);
