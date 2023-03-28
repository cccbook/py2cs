# 用 ShortGPT 寫軟體工程書籍

```
ccckmit@asus MINGW64 /d/ccc/py2cs/_書/軟體工程 (master)
$ shortgpt.sh
Welcome to shortgpt. You may use the following commands
1. quit
2. history
3. shell <command>
4. chat <prompt>
5. fchat <file> <prompt>

You may use the following $key for short
{
  "mt": "翻譯下列文章",
  "tw": "以 繁體中文 格式輸出",
  "en": "output in English",
  "jp": "output in Japanese",
  "md": "format in Markdown+LaTex, add space before and after $..$"
}

command> fchat 0.md 請寫一本軟體工程的書，先寫目錄
========question=======
請寫一本軟體工程的書，先寫目錄
========response=======
Response will write to file:0.md

command> fchat 0.md 請寫一本軟體工程的書，先寫目錄，章節必須標上 1.1 這樣的開頭
========question=======
請寫一本軟體工程的書，先寫目錄，章節必須標上 1.1 這樣的開頭
========response=======
Response will write to file:0.md

command> fchat 1.md 第一章：軟體工程基礎
========question=======
第一章：軟體工程基礎
========response=======
Response will write to file:1.md

command> fchat 1.1.md 1.1 軟體工程概述
========question=======
1.1 軟體工程概述
========response=======
Response will write to file:1.1.md

command> fchat 1.2.md 1.2 軟體開發生命週期
========question=======
1.2 軟體開發生命週期
========response=======
Response will write to file:1.2.md

command> fchat 1.3.md 1.3 需求分析
========question=======
1.3 需求分析
========response=======
Response will write to file:1.3.md

command> fchat 1.4.md 1.4 軟體設計
========question=======
1.4 軟體設計
========response=======
Response will write to file:1.4.md

command> fchat 1.5.md 1.5 軟體測試
========question=======
1.5 軟體測試
========response=======
Response will write to file:1.5.md

command> fchat 1.6.md 1.6 軟體維護
========question=======
1.6 軟體維護
========response=======
Response will write to file:1.6.md

command> fchat 2.md 第二章：程式設計
========question=======
第二章：程式設計
========response=======
Response will write to file:2.md

command> fchat 2.1.md 2.1 程式設計原則
========question=======
2.1 程式設計原則
========response=======
Response will write to file:2.1.md

command> fchat 2.2.md 2.2 物件導向程式設計
========question=======
2.2 物件導向程式設計
========response=======
Response will write to file:2.2.md

command> fchat 2.3.md 2.3 資料結構
========question=======
2.3 資料結構
========response=======
Response will write to file:2.3.md

command> fchat 2.4.md 2.4 演算法
========question=======
2.4 演算法
========response=======
Response will write to file:2.4.md

command> fchat 3.md 第三章：軟體開發方法論
========question=======
第三章：軟體開發方法論
========response=======
Response will write to file:3.md

command> fchat 3.1.md 3.1 瀑布式開發
========question=======
3.1 瀑布式開發
========response=======
Response will write to file:3.1.md

command> fchat 3.2.md 3.2 敏捷式開發
========question=======
3.2 敏捷式開發
========response=======
Response will write to file:3.2.md

command> fchat 3.3.md 3.3 DevOps
========question=======
3.3 DevOps
========response=======
Response will write to file:3.3.md

command> fchat 4.md 第四章：版本控制
========question=======
第四章：版本控制
========response=======
Response will write to file:4.md

command> fchat 4.1.md 4.1 版本控制概述
========question=======
4.1 版本控制概述
========response=======
Response will write to file:4.1.md

command> fchat 4.2.md 4.2 Git基礎
========question=======
4.2 Git基礎
========response=======
Response will write to file:4.2.md

command> fchat 4.3.md 4.3 Git進階
========question=======
4.3 Git進階
========response=======
Response will write to file:4.3.md

command> fchat 5.md 第五章：軟體測試
========question=======
第五章：軟體測試
========response=======
Response will write to file:5.md

command> fchat 5.1.md 5.1 測試方法論
========question=======
5.1 測試方法論
========response=======
Response will write to file:5.1.md

command> fchat 5.2.md 5.2 單元測試
========question=======
5.2 單元測試
========response=======
Response will write to file:5.2.md

command> fchat 5.3.md 5.3 整合測試
========question=======
5.3 整合測試
========response=======
Response will write to file:5.3.md

command> fchat 5.4.md 5.4 系統測試
========question=======
5.4 系統測試
========response=======
Response will write to file:5.4.md

command> fchat 6.md 第六章：軟體品質管理
========question=======
第六章：軟體品質管理
========response=======
Response will write to file:6.md

command> fchat 6.1.md 6.1 品質管理概念
========question=======
6.1 品質管理概念
========response=======
Response will write to file:6.1.md

command> fchat 6.2.md 6.2 軟體品質指標
========question=======
6.2 軟體品質指標
========response=======
Response will write to file:6.2.md

command> fchat 6.3.md 6.3 品質管制工具
========question=======
6.3 品質管制工具
========response=======
Response will write to file:6.3.md

command> fchat 7.md 第七章：軟體安全
========question=======
第七章：軟體安全
========response=======
Response will write to file:7.md

command> fchat 7.1.md 7.1 安全原則
========question=======
7.1 安全原則
========response=======
Response will write to file:7.1.md

command> fchat 7.2.md 7.2 安全設計
========question=======
7.2 安全設計
========response=======
Response will write to file:7.2.md

command> fchat 7.3.md 7.3 安全測試
========question=======
7.3 安全測試
========response=======
Response will write to file:7.3.md

command> fchat 7.4.md 7.4 安全漏洞修補
========question=======
7.4 安全漏洞修補
========response=======
Response will write to file:7.4.md

command> fchat 8.md 第八章：專案管理
========question=======
第八章：專案管理
========response=======
Response will write to file:8.md

command> fchat 8.1.md 8.1 專案管理概述
========question=======
8.1 專案管理概述
========response=======
Response will write to file:8.1.md

command> fchat 8.2.md 8.2 專案範疇管理
========question=======
8.2 專案範疇管理
========response=======
Response will write to file:8.2.md

command> fchat 8.3.md 8.3 專案時間管理
========question=======
8.3 專案時間管理
========response=======
Response will write to file:8.3.md

command> fchat 8.4.md 8.4 專案成本管理
========question=======
8.4 專案成本管理
========response=======
Response will write to file:8.4.md

command> quit

ccckmit@asus MINGW64 /d/ccc/py2cs/_書/軟體工程 (master)
$ ls
0.md    1.5.md  2.3.md  3.3.md  4.md    5.md    7.1.md  8.1.md  ShortGpt.md
1.1.md  1.6.md  2.4.md  3.md    5.1.md  6.1.md  7.2.md  8.2.md
1.2.md  1.md    2.md    4.1.md  5.2.md  6.2.md  7.3.md  8.3.md
1.3.md  2.1.md  3.1.md  4.2.md  5.3.md  6.3.md  7.4.md  8.4.md
1.4.md  2.2.md  3.2.md  4.3.md  5.4.md  6.md    7.md    8.md
```
