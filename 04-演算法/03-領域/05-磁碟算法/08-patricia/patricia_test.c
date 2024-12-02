#include "patricia.h"

int main() {
    char text[] = "abcabababacba";
    // char text[] = "居里夫人:弱者坐待時機 強者製造時機";
    int n = strlen(text);
    patNode *root = patInsert(NULL, text, 1*8);
    for (char *p=text; *p; p++) {
        patInsert(root, p, 8*min(4, text+n-p));
    }
    patNode *node = patSearch(root, "ab", 2*8);
    printf("search(ab,2):"); patPrint(node);
    printf("========== dump ==============\n");
    patDump(root, 0);
}
