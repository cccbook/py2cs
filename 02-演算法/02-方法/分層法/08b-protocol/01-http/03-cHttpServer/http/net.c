#include "net.h"

char ip[SMAX];

char *host_to_ip(char *hostname, char *ip) { // 查出 host 對應的 ip
  struct hostent *host = gethostbyname(hostname);
  inet_ntop(host->h_addrtype, host->h_addr_list[0], ip, SMAX); // 取第一個 IP
  return ip;
}

int net_init(net_t *net, int protocol, int side, int port, char *host) {
  memset(net, 0, sizeof(net_t));
  net->protocol = protocol;
  net->side = side;
  net->port = port;
  net->serv_ip = (side==CLIENT) ? host_to_ip(host, ip) : "127.0.0.1";
  int socketType = (protocol == TCP) ? SOCK_STREAM : SOCK_DGRAM;
  net->sock_fd = socket(AF_INET, socketType, 0);
  assert(net->sock_fd >= 0);
  net->serv_addr.sin_family = AF_INET;
  net->serv_addr.sin_addr.s_addr = (side == SERVER) ? htonl(INADDR_ANY) : inet_addr(net->serv_ip);
  net->serv_addr.sin_port = htons(net->port);
  return 0;
}

int net_connect(net_t *net) {
  int r = connect(net->sock_fd, (struct sockaddr *)&net->serv_addr, sizeof(net->serv_addr));
  // assert(r>=0);
  return r;
}

int net_bind(net_t *net) {
  int r = bind(net->sock_fd, (struct sockaddr*)&net->serv_addr, sizeof(net->serv_addr));
  assert(r>=0);
  return r;
}

int net_listen(net_t *net) {
  int r = listen(net->sock_fd, 10); // 最多十個同時連線
  assert(r>=0);
  return r;
}

int net_accept(net_t *net) {
  int r = accept(net->sock_fd, (struct sockaddr*)NULL, NULL);
  assert(r>=0);
  return r;
}

int net_close(net_t *net) {
  shutdown(net->sock_fd, SHUT_WR);
  close(net->sock_fd);
  return 0;
}
