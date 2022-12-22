# csvToSqlite.py

## 執行

執行完下列程式後，會產生一個 stockPrice.db 檔案，這是 sqlite 的資料庫檔案，可以用 sqliteExpert 去開啟並檢視管理資料。

```
$ python csvToSqlite.py
日期 開盤價 收盤價 成交筆數
111/10/03 418.50 417.00 70127
command= INSERT INTO stockPrice (日期, 開盤價, 收盤價, 成交筆數) VALUES("111/10/03", 418.50, 417.00, 70127) 
111/10/04 429.50 429.00 31696
command= INSERT INTO stockPrice (日期, 開盤價, 收盤價, 成交筆數) VALUES("111/10/04", 429.50, 429.00, 31696) 
111/10/05 444.50 445.00 51865
command= INSERT INTO stockPrice (日期, 開盤價, 收盤價, 成交筆數) VALUES("111/10/05", 444.50, 445.00, 51865) 
111/10/06 450.50 451.00 24907
command= INSERT INTO stockPrice (日期, 開盤價, 收盤價, 成交筆數) VALUES("111/10/06", 450.50, 451.00, 24907) 
111/10/07 448.50 438.00 45112
command= INSERT INTO stockPrice (日期, 開盤價, 收盤價, 成交筆數) VALUES("111/10/07", 448.50, 438.00, 45112) 
111/10/11 408.00 401.50 210614
command= INSERT INTO stockPrice (日期, 開盤價, 收盤價, 成交筆數) VALUES("111/10/11", 408.00, 401.50, 210614)
111/10/12 397.00 397.50 114055
command= INSERT INTO stockPrice (日期, 開盤價, 收盤價, 成交筆數) VALUES("111/10/12", 397.00, 397.50, 114055)
111/10/13 400.50 395.00 66846
command= INSERT INTO stockPrice (日期, 開盤價, 收盤價, 成交筆數) VALUES("111/10/13", 400.50, 395.00, 66846) 
111/10/14 414.50 412.00 54666
command= INSERT INTO stockPrice (日期, 開盤價, 收盤價, 成交筆數) VALUES("111/10/14", 414.50, 412.00, 54666) 
111/10/17 400.00 397.00 66589
command= INSERT INTO stockPrice (日期, 開盤價, 收盤價, 成交筆數) VALUES("111/10/17", 400.00, 397.00, 66589)
111/10/18 406.50 407.00 30538
command= INSERT INTO stockPrice (日期, 開盤價, 收盤價, 成交筆數) VALUES("111/10/18", 406.50, 407.00, 30538) 
111/10/19 404.00 395.50 68211
command= INSERT INTO stockPrice (日期, 開盤價, 收盤價, 成交筆數) VALUES("111/10/19", 404.00, 395.50, 68211) 
111/10/20 391.00 397.50 76430
command= INSERT INTO stockPrice (日期, 開盤價, 收盤價, 成交筆數) VALUES("111/10/20", 391.00, 397.50, 76430) 
111/10/21 395.50 389.50 64043
command= INSERT INTO stockPrice (日期, 開盤價, 收盤價, 成交筆數) VALUES("111/10/21", 395.50, 389.50, 64043) 

```

## sqliteExpert

* https://www.sqliteexpert.com/
    * 下載 -- https://www.sqliteexpert.com/v5/SQLiteExpertPersSetup64.exe

安裝好後可以用來開啟 stockPrice.db 檔案，然後下些指令觀察。

> SELECT * FROM stockPrice

![](img/sqliteExpert1.png)

> SELECT * FROM stockPrice ORDER BY 成交筆數 DESC

![](img/sqliteExpert1.png)