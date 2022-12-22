LIBS = http/httpd.c http/net.c
OBJS = cServer
OPTS = -Wall -D_DEBUG

all: $(OBJS)

cServer:
	gcc $(OPTS) $(LIBS) cServer.c -lpthread -o cServer

clean:
	rm $(OBJS)
