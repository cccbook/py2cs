# UTF8

* https://www.cprogramming.com/tutorial/unicode.html

UTF-8 is a specific scheme for mapping a sequence of 1-4 bytes to a number from 0x000000 to 0x10FFFF:

```
00000000 -- 0000007F: 	0xxxxxxx
00000080 -- 000007FF: 	110xxxxx 10xxxxxx
00000800 -- 0000FFFF: 	1110xxxx 10xxxxxx 10xxxxxx
00010000 -- 001FFFFF: 	11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
```