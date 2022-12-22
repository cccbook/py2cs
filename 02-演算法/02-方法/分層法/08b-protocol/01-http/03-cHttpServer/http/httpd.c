#include "httpd.h"

void readRequest(int client, char *request) {
  int len = read(client, request, TMAX);
  request[len] = '\0';
}

void parseRequest(char *request, char *op, char *path, char *body) {
  sscanf(request, "%s %s HTTP/", op, path);
  char *p = strstr(request, "\r\n\r\n");
  if (p) strcpy(body, p+4); else *body='\0';
}

void responseText(int client, int status, char *text) {
  char response[SMAX];
  sprintf(response, "HTTP/1.1 %d OK\r\n"
                    "Content-Type: text/plain; charset=UTF-8\r\n"
                    "Content-Length: %ld\r\n\r\n", status, strlen(text));
  write(client, response, strlen(response));
  write(client, text, strlen(text));
}

void responseFile(int client, char *path) {
  char text[TMAX-200], response[TMAX], fpath[SMAX];
  sprintf(fpath, "./web%s", path); // ex: fpath = ./web/hello.html
  printf("responseFile:fpath=%s\n", fpath);
  FILE *file = fopen(fpath, "r");
  int len;
  if (file == NULL) {
    strcpy(text, "<html><body><h1>File not Found!</h1></body></html>");
    len = strlen(text);
  } else {
    len = fread(text, 1, TMAX, file);
    text[len] = '\0';
  }
  sprintf(response, "HTTP/1.1 200 OK\r\n"
                    "Content-Type: text/html; charset=UTF-8\r\n"
                    "Content-Length: %d\r\n\r\n%s", len, text);
  write(client, response, strlen(response));
}
