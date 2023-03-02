#include <pthread.h>
#include "http/net.h"
#include "http/httpd.h"

void *serve(void *argu) {
  int client = *(int*) argu;
  if (client == -1) {
    printf("Can't accept");
    return NULL;
  }
  char request[TMAX], path[SMAX], op[SMAX], body[TMAX];
  readRequest(client, request);
  printf("===========request=============\n%s\n", request);
  parseRequest(request, op, path, body);
  printf("op=%s path=%s body=%s\n", op, path, body);
  if (strcmp(path, "/")==0)
    responseText(client, 200, "success");
  else
    responseText(client, 404, "fail");
  sleep(1);
  close(client);
  return NULL;
}

int main(int argc, char *argv[]) {
  int port = (argc >= 2) ? atoi(argv[1]) : PORT;
  net_t net;
  net_init(&net, TCP, SERVER, port, NULL);
  net_bind(&net);
  net_listen(&net);
  printf("Server started at: http://127.0.0.1:%d\n", net.port);
  while (1) {
    int client = net_accept(&net);
    pthread_t thread1;
    pthread_create(&thread1, NULL, &serve, &client);
  }
}
