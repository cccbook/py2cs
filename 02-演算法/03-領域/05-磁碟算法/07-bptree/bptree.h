#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>
#include <assert.h>

#define NCHILD 120

#ifdef __DISK__
#define NODE int
#else
#define NODE struct bptNode*
#endif

typedef struct bptNode {
	struct {
		bool isLeaf : 1;
		bool inDisk : 1;
	};
	uint16_t nkey;
	union {
		int key[NCHILD];
		char *skey[NCHILD];
	};
	void* child[NCHILD]; // 樹葉的 child 指向 value，中間指向 bptNode
	NODE father;
	NODE next;
	NODE last;
} bptNode;

// extern void bptSetMaxChildNumber(int);
extern void bptInit();
extern void bptDestroy();
extern int bptInsert(int, void*);
extern int bptGetTotalNodes();
extern int bptQueryKey(int);
extern int bptQueryRange(int, int);
extern void bptModify(int, void*);
extern void bptDelete(int);
extern int bptFind(int key);
