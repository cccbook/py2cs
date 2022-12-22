#pragma once
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// Disk layout:
// [ super block | inode blocks | free bit map | data blocks]
//
typedef uint32_t uint;
typedef uint8_t  byte;
// On-disk file system format. // 磁碟中的檔案系統結構
// Both the kernel and user programs use this header file.

#define NBLOCK       1000  // 最大區塊數
#define MAXPATH      128   // 最大路徑長度

#define ROOTINO  1   // root i-number 根結點
#define BSIZE 1024   // block size    區塊大小    

// mkfs computes the super block and builds an initial file system. The
// super block describes the disk layout:
struct superblock {  // 超級區塊
  uint magic;        // Must be FSMAGIC      // 用來辨識的魔數:0x10203040
  uint blocks;       // Size of file system image (blocks) // 全部區塊數
  uint dblocks;      // Number of data blocks // 資料區塊數
  uint inodes;       // Number of inodes.     // inodes 數量
  uint istart;       // Block number of first inode block // inode 的首區塊
  uint bstart;       // Block number of first free map block // free bitmap 的首區塊
};

#define FSMAGIC 0x10203040

#define NDIRECT 14
#define NINDIRECT (BSIZE / sizeof(uint))
#define MAXFILE (NDIRECT + NINDIRECT)

struct inode {
  uint size;          // 檔案大小
  uint addrs[NDIRECT+1]; // 區塊位址
};

// Inodes per block. (每個區塊的 inode 數量)
#define IPB           (BSIZE / sizeof(struct inode))

// Block containing inode i (第 i 個 inode 存放在哪個區塊)
#define IBLOCK(i, sb)     ((i) / IPB + sb.istart)

// Bitmap bits per block (每個區塊包含幾個 bits)
#define BPB           (BSIZE*8)

// Block of free map containing bit for block b (取得 free map 中紀錄該區塊是否 free 的區塊代號)
#define BBLOCK(b, sb) ((b)/BPB + sb.bstart)

// Directory is a file containing a sequence of dirent structures.
#define DIRSIZ 14

struct dirent { // 目錄中的一項 (inode 代號+名稱)。
  uint inum;
  char name[DIRSIZ];
};

struct buf { // 緩衝區塊
  int valid;    // has data been read from disk? // 已讀入？
  int disk;     // does disk "own" buf? // 正在讀？
  uint dev;     // 裝置代號
  uint blockno; // 區塊代號
  // struct sleeplock lock; // 鎖定中？
  uint refcnt;      // 引用數量
  struct buf *prev; // LRU cache list
  struct buf *next; // 緩衝區塊串列
  byte data[BSIZE]; // 資料內容
};

// ============ stat.h =================
enum {T_DIR, T_FILE}; // Directory/File

struct stat { // 狀態資訊，可能是《目錄，檔案或裝置》
  uint ino;    // Inode number
  short type;  // Type of file
  uint size; // Size of file in bytes
};

FILE *fs_create(char *fname);
FILE *fs_open(char *fname);