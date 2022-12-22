#include <stdio.h>

struct node {
  char *data;
  struct node *next;
};

struct node q[10] = {
  {.data="a", .next=&q[1]},
  {.data="b", .next=&q[2]},
  {.data="c", .next=&q[3]},
  {.data="d", .next=NULL},
};

struct queue {
  struct node *front, *rear;
};

struct queue Q = {
  .front = &q[0],
  .rear = &q[3],
};

int main() {
  for (struct node *p=Q.front; p!=NULL; p=p->next) {
    printf("%s->", p->data);
  }
  printf("NIL\n");
}