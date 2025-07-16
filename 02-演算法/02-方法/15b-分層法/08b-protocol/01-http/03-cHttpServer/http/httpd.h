#ifndef __HTTPD_H__
#define __HTTPD_H__

#include "net.h"

void readRequest(int client, char *request);
void parseRequest(char *request, char *op, char *path, char *body);
void responseText(int client, int status, char *text);
void responseFile(int client, char *path);

#endif
