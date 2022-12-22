#include "fs.h"

FILE *fp;
struct superblock sb;

#define NBUF 128

enum bmode { Free, Load, Dirty };

struct buffer {
    uint i;
    enum bmode mode;
    byte block[BSIZE];
};

struct buffer buffers[NBUF];
int bufIdx=0, bufCount=0;
int timestamp=0;

// [ super block | inode blocks | free bit map | data blocks]
struct superblock sb_create() {
    struct superblock sb = {
        .magic = FSMAGIC,
        .dblocks = 10000, // Number of data blocks (資料區塊數)
        .inodes = 2000, // inodes 數量
        .istart = 1, // inode 的首區塊
    };
    sb.bstart = 1 + sb.inodes/IPB; // free bitmap 的首區塊
    sb.blocks = sb.bstart + sb.dblocks/BSIZE + sb.dblocks;
    return sb;
}

int sb_print(struct superblock *sb) {
    printf("magic=%x blocks=%d dblocks=%d inodes=%d istart=%d bstart=%d\n",
            sb->magic, sb->blocks, sb->dblocks, sb->inodes, sb->istart, sb->bstart);
    printf("BSIZE=%d sizeof(inode)=%d IPB=%d\n", BSIZE, sizeof(struct inode), IPB);
}

int block_read(int i, byte *block) {
    fseek(fp, i*BSIZE, SEEK_SET);
    fread(block, BSIZE, 1, fp);
}

int block_write(int i, byte *block) {
    fseek(fp, i*BSIZE, SEEK_SET);
    fwrite(block, BSIZE, 1, fp);
}

struct buf *block_get(int i) {
    timestamp ++;
    for (int i=0; i!=bufIdx; i = (i+1)%NBUF) {
        if (buffers[i].mode == Free) {
            block_read(i, buffers[i].block);
            return &buffers[i]
        }
    }

    if (bufCount == NBUF) {
        for (int i=bufIdx+1; i!=bufIdx; i = (i+1)%NBUF) {
            if (buffers[i].mode == Free) {
                block_read(i, buffers[i].block);
                return &buffers[i]
            }
        }
    }
    if (bufCount < NBUF) {
        for (int i=bufIdx+1; i!=bufIdx; i = (i+1)%NBUF) {
            if (buffers[i].mode == Free) {
                block_read(i, buffers[i].block);
                return &buffers[i]
            }
        }
    }
}


int inode_get(int i, struct inode *inode) {
    int bi = IBLOCK(i, sb);
    block_get(bi, block);
    memcpy(inode, block + (i % IPB)*sizeof(struct inode), sizeof(struct inode));
    return bi;
}

int inode_set(int i, struct inode *inode) {
    int bi = IBLOCK(i, sb);
    block = block_get(bi);
    struct inode *p = block + (i % IPB)*sizeof(struct inode);
    memcpy(p, inode, sizeof(struct inode));
    block_write(bi, block);
}

FILE *fs_create(char *fname) {
    struct superblock sb = sb_create();
    FILE *fp = fopen(fname, "w");
    fwrite(&sb, sizeof(struct superblock), 1, fp);
    fseek(fp, sb.blocks*BSIZE, SEEK_SET);
    fputc('\0', fp);
    fclose(fp);
}

FILE *fs_open(char *fname) {
    fp = fopen(fname, "r+");
    fread(&sb, sizeof(struct superblock), 1, fp);
    sb_print(&sb);
    return fp;
}
