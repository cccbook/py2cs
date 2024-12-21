// ref -- https://github.com/json-db/B-Plus-Tree
#include "bptree.h"

struct bptNode* Root;

int TotalNodes;

/** Create a new B+tree Node */
bptNode* newNode() {
	struct bptNode* p = (struct bptNode*)malloc(sizeof(struct bptNode));
	assert(p);
	p->isLeaf = false;
	p->nkey = 0;
	p->child[0] = NULL;
	p->father = NULL;
	p->next = NULL;
	p->last = NULL;
	TotalNodes++;
	return p;
}

/** Binary search to find the biggest child l that node->key[l] <= key */
// inline int find(bptNode* node, int key) {
int find(bptNode* node, int key) {
	int l = 0, r = node->nkey;
	if (key < node->key[l]) return l;
	if (node->key[r - 1] <= key) return r - 1;
	while (l < r - 1) {
		int mid = (l + r) >> 1;
		if (node->key[mid] > key)
			r = mid;
		else
			l = mid;
	}
	return l;
}

/**
 * node(NCHILD) split into two part:
 *	(1) node(0 .. Mid - 1) with original key
 *	(2) Temp(Mid .. NCHILD) with key[Mid]
 * where Mid = NCHILD / 2
 * Note that only when split() is called, a new Node is created
 */
void insert(bptNode*, int, void*);
void split(bptNode* node) {
	// copy node(Mid .. NCHILD) -> Temp(0 .. Temp->nkey)
	bptNode* Temp = newNode();
	bptNode* ch;
	int Mid = NCHILD >> 1;
	Temp->isLeaf = node->isLeaf; // Temp's depth == node's depth
	Temp->nkey = NCHILD - Mid;
	int i;
	for (i = Mid; i < NCHILD; i++) {
		Temp->child[i - Mid] = node->child[i];
		Temp->key[i - Mid] = node->key[i];
		if (!Temp->isLeaf) {
			ch = (bptNode*)Temp->child[i - Mid];
			ch->father = Temp;
		}
	}
	// Change node
	node->nkey = Mid;
	// insert Temp
	if (node == Root) {
		// Create a new Root, the depth of Tree is increased
		Root = newNode();
		Root->nkey = 2;
		Root->key[0] = node->key[0];
		Root->child[0] = node;
		Root->key[1] = Temp->key[0];
		Root->child[1] = Temp;
		node->father = Temp->father = Root;
		if (node->isLeaf) {
			node->next = Temp;
			Temp->last = node;
		}
	} else {
		// Try to insert Temp to node->father
		Temp->father = node->father;
		insert(node->father, node->key[Mid], (void*)Temp);
	}
}

/** insert (key, value) into node, if node is full, then split it to fit the definition of B+tree */
void insert(bptNode* node, int key, void* value) {
	int i, ins;
	if (key < node->key[0]) ins = 0; else ins = find(node, key) + 1;
	for (i = node->nkey; i > ins; i--) {
		node->key[i] = node->key[i - 1];
		node->child[i] = node->child[i - 1];
	}
	node->nkey++;
	node->key[ins] = key;
	node->child[ins] = value;
	if (node->isLeaf == false) { // make links on leaves
		bptNode* firstChild = (bptNode*)(node->child[0]);
		if (firstChild->isLeaf == true) { // which means value is also a leaf as child[0]	
			bptNode* temp = (bptNode*)(value);
			if (ins > 0) {
				bptNode* prevChild;
				bptNode* succChild;
				prevChild = (bptNode*)node->child[ins - 1];
				succChild = prevChild->next;
				prevChild->next = temp;
				temp->next = succChild;
				temp->last = prevChild;
				if (succChild != NULL) succChild->last = temp;
			} else {
				// do not have a prevChild, then refer next directly
				// updated: the very first record on B+tree, and will not come to this case
				temp->next = node->child[1];
				printf("this happens\n");
			}
		}
	}
	if (node->nkey == NCHILD) // children is full
		split(node);
}

/** resort(Give, Get) make their no. of children average */
void resort(bptNode* Left, bptNode* Right) {
	int total = Left->nkey + Right->nkey;
	bptNode* temp;
	if (Left->nkey < Right->nkey) {
		int leftSize = total >> 1;
		int i = 0, j = 0;
		while (Left->nkey < leftSize) {
			Left->key[Left->nkey] = Right->key[i];
			Left->child[Left->nkey] = Right->child[i];
			if (!Left->isLeaf) {
				temp = (bptNode*)(Right->child[i]);
				temp->father = Left;
			}
			Left->nkey++;
			i++;
		}
		while (i < Right->nkey) {
			Right->key[j] = Right->key[i];
			Right->child[j] = Right->child[i];
			i++;
			j++;
		}
		Right->nkey = j; 
	} else {
		int leftSize = total >> 1;
		int i, move = Left->nkey - leftSize, j = 0;
		for (i = Right->nkey - 1; i >= 0; i--) {
			Right->key[i + move] = Right->key[i];
			Right->child[i + move] = Right->child[i];
		}
		for (i = leftSize; i < Left->nkey; i++) {
			Right->key[j] = Left->key[i];
			Right->child[j] = Left->child[i];
			if (!Right->isLeaf) {
				temp = (bptNode*)Left->child[i];
				temp->father = Right;
			}
			j++;
		}
		Left->nkey = leftSize;
		Right->nkey = total - leftSize;
	}
}

/**
 * redistribute node, using following strategy:
 * (1) resort with right brother
 * (2) resort with left brother
 * (3) merge with right brother
 * (4) merge with left brother
 * in that case root has only one child, set this chil to be root
 */
void delete(bptNode*, int);
void redistribute(bptNode* node) {
	if (node == Root) {
		if (node->nkey == 1 && !node->isLeaf) {
			Root = node->child[0];
			free(node);
		}
		return;
	}
	bptNode* father = node->father;
	bptNode* prevChild;
	bptNode* succChild;
	bptNode* temp;
	int ipos = find(father, node->key[0]);
	if (ipos + 1 < father->nkey) {
		succChild = father->child[ipos + 1];
		if ((succChild->nkey - 1) * 2 >= NCHILD) { // at least can move one child to node
			resort(node, succChild); // (1) resort with right child
			father->key[ipos + 1] = succChild->key[0];
			return;
		}
	}
	if (ipos - 1 >= 0) {
		prevChild = father->child[ipos - 1];
		if ((prevChild->nkey - 1) * 2 >= NCHILD) {
			resort(prevChild, node); // (2) resort with left child
			father->key[ipos] = node->key[0];
			return;
		}
	}
	if (ipos + 1 < father->nkey) { // (3) merge with right child
		int i = 0;
		while (i < succChild->nkey) {
			node->key[node->nkey] = succChild->key[i];
			node->child[node->nkey] = succChild->child[i];
			if (!node->isLeaf) {
				temp = (bptNode*)(succChild->child[i]);
				temp->father = node;
			}
			node->nkey++;
			i++;
		}
		delete(father, succChild->key[0]); // delete right child
		return;
	}
	if (ipos - 1 >= 0) { // (4) merge with left child
		int i = 0;
		while (i < node->nkey) {
			prevChild->key[prevChild->nkey] = node->key[i];
			prevChild->child[prevChild->nkey] = node->child[i];
			if (!node->isLeaf) {
				temp = (bptNode*)(node->child[i]);
				temp->father = prevChild;
			}
			prevChild->nkey++;
			i++;
		}
		delete(father, node->key[0]); // delete left child
		return;
	}
	printf("What?! you're the only child???\n"); // this won't happen
}

/** delete key from node, if no. of children < MaxChildNUmber / 2, resort or merge it with brothers */
void delete(bptNode* node, int key) {
	int i, del = find(node, key);
	void* delChild = node->child[del];
	for (i = del; i < node->nkey - 1; i++) {
		node->key[i] = node->key[i + 1];
		node->child[i] = node->child[i + 1];
	}
	node->nkey--;
	if (node->isLeaf == false) { // make links on leaves
		bptNode* firstChild = (bptNode*)(node->child[0]);
		if (firstChild->isLeaf == true) { // which means delChild is also a leaf
			bptNode* temp = (bptNode*)delChild;
			bptNode* prevChild = temp->last;
			bptNode* succChild = temp->next;
			if (prevChild != NULL) prevChild->next = succChild;
			if (succChild != NULL) succChild->last = prevChild;
		}
	}
	if (del == 0 && node!=Root) { // some fathers' key should be changed
		bptNode* temp = node;
		while (temp != Root && temp == temp->father->child[0]) {
			temp->father->key[0] = node->key[0];
			temp = temp->father;
		}
		if (temp != Root) {
			temp = temp->father;
			int i = find(temp, key);
			temp->key[i] = node->key[0];
		}
	}
	free(delChild);
	if (node->nkey * 2 < NCHILD)
		redistribute(node);
}

/** findLeaf a leaf node that key lays in it
 *	modify indicates whether key should affect the tree
 */
bptNode* findLeaf(int key, int modify) {
	bptNode* node = Root;
	while (1) {
		if (node->isLeaf == true)
			break;
		if (key < node->key[0]) {
			if (modify == true) node->key[0] = key;
			node = node->child[0];
		} else {
			int i = find(node, key);
			node = node->child[i];
		}
	}
	return node;
}

/** destroy subtree whose root is node, By recursion */
void destroy(bptNode* node) {
	if (node->isLeaf == true) {
		int i;
		for (i = 0; i < node->nkey; i++)
			free(node->child[i]);
	} else {
		int i;
		for (i = 0; i < node->nkey; i++)
			destroy(node->child[i]);
	}
	free(node);
}

/** print subtree whose root is node */
void print(bptNode* node) {
	int i;
	for (i = 0; i < node->nkey; i++)
		printf("%d ", node->key[i]);
	printf("\n");
	if (!node->isLeaf) {
		for (i = 0; i < node->nkey; i++)
			print(node->child[i]);
	}
}

/** Interface: insert (key, value) into B+tree */
int bptInsert(int key, void* value) {
	bptNode* leaf = findLeaf(key, true);
	int i = find(leaf, key);
	// if (leaf->key[i] == key) return false; // key 重複就不新增 ???
	insert(leaf, key, value);
	return true;
}

/** Interface: query all record whose key satisfy that key = querykey */
int bptQueryKey(int key) {
	bptNode* leaf = findLeaf(key, false);
	int ansNum = 0;
	int i;
	for (i = 0; i < leaf->nkey; i++) {
		//printf("%d ", leaf->key[i]);
		if (leaf->key[i] == key) {
			ansNum++;
			if (ansNum < 20) printf("[no.%d	key = %d, value = %s]\n", ansNum, leaf->key[i], (char*)leaf->child[i]);
		}
	}
	printf("Total number of answers is: %d\n", ansNum);
	return ansNum;
}

/** Interface: query all record whose key satisfy that queryl <= key <= queryr */
int bptQueryRange(int l, int r) {
	bptNode* leaf = findLeaf(l, false);
	int ansNum = 0;
	int i;
	for (i = 0; i < leaf->nkey; i++) {
		//printf("%d ", leaf->key[i]);
		if (leaf->key[i] >= l) break;
	}
	int finish = false;
	while (!finish) {
		while (i < leaf->nkey) {
			if (leaf->key[i] > r) {
				finish = true;
				break;
			}
			ansNum++;
			if (ansNum == 20) printf("...\n");
			if (ansNum < 20) printf("[no.%d	key = %d, value = %s]\n", ansNum, leaf->key[i], (char*)leaf->child[i]);
			i++;
		}
		if (finish || leaf->next == NULL) break;
		leaf = leaf->next;
		i = 0;
	}
	printf("Total number of answers is: %d\n", ansNum);
	return ansNum;
}

/** Interface: modify value on the given key */
void bptModify(int key, void* value) {
	bptNode* leaf = findLeaf(key, false);
	int i = find(leaf, key);
	if (leaf->key[i] != key) return; // don't have this key
	printf("Modify: key = %d, original value = %s, new value = %s\n", key, (char*)(leaf->child[i]), (char*)(value));
	free(leaf->child[i]);
	leaf->child[i] = value;
}

/** Interface: delete value on the given key */
void bptDelete(int key) {
	bptNode* leaf = findLeaf(key, false);
	int i = find(leaf, key);
	if (leaf->key[i] != key) return; // don't have this key
	printf("delete: key = %d, original value = %s\n", key, (char*)(leaf->child[i]));
   	delete(leaf, key); 
}

/** Interface: Called to destroy the B+tree */
void bptDestroy() {
	if (Root == NULL) return;
	printf("Now destroying B+tree ..\n");
	destroy(Root);
	Root = NULL;
	printf("Done.\n");
}

void bptInit() {
	bptDestroy();
	Root = newNode();
	Root->isLeaf = true;
	TotalNodes = 0;
}

void bptPrint() {
	struct bptNode* leaf = findLeaf(1000000000, false);
	int cnt = 0;
	while (leaf != NULL) {
		int i;
		for (i = leaf->nkey - 1; i >= 0; i--) {
			printf("%4d ", leaf->key[i]);
			if (++cnt % 20 == 0) printf("\n");
		}
		leaf = leaf->last;
	}
}
