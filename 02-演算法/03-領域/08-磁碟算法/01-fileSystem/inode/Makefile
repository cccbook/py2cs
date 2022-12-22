CC = gcc
CFLAGS = 
all: mkfs testfs

mkfs: mkfs.c fs.c
	$(CC) $(CFLAGS) -o $@ $^

testfs: 
	./mkfs fs.img README.md fs.c fs.h

clean:
	rm -f *.exe
