#include <stdio.h>
#include <string.h>
#include "utf8.h"

int main() {
    char line[] = "Hello 你好!\n";
    printf("line=%s\n", line);
    char *s = strstr(line, "你");
    printf("s=%s\n", s);
    for (char *p=line; *p;) {
        int len = utf8len(p);
        printf("len=%d:%.*s\n", len, len, p);
        p+=len;
    }
}