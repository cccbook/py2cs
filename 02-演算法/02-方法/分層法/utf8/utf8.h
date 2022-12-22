/*

UTF-8 is a specific scheme for mapping a sequence of 1-4 bytes to a number from 0x000000 to 0x10FFFF:

00000000 -- 0000007F: 	0xxxxxxx
00000080 -- 000007FF: 	110xxxxx 10xxxxxx
00000800 -- 0000FFFF: 	1110xxxx 10xxxxxx 10xxxxxx
00010000 -- 001FFFFF: 	11110xxx 10xxxxxx 10xxxxxx 10xxxxxx

常見中文位於 \u4e00 到 \u9fff 之間，佔 3 bytes
*/

#define utf8len(s) ((s[0]&0xF0)==0xF0?4:\
                    (s[0]&0xE0)==0xE0?3:\
                    (s[0]&0xC0)==0xC0?2:\
                    1)

// [C语言与中文的一些测试 (Win, UTF8源码)](https://zhuanlan.zhihu.com/p/71778196)
#ifdef _WIN32
#define utf8init() system("chcp 65001")
#else
#define utf8init() {}
#endif
