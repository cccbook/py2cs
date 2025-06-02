

https://www.facebook.com/modeerf/posts/pfbid02sNW8T6Vyac39Zm1MjNDqohnd47NJFQ6WtPX57Du4GmBfiYSzTWD4nNzCS8cUCNsgl

Cursor 如何地端檢索龐大的 code？

⸻

什麼是「Codebase Indexing」？
➤ Cursor 會把你整個專案掃過一遍，做成可語意搜尋的索引
 • 一開啟資料夾就自動開啟此功能（可在設定關閉）。
 • 之後你問 AI 問題時，它就能帶著「整個程式碼庫的脈絡」來回答或自動補 Code。

⸻

掃描與上傳流程

➤ 先算 Merkle Tree
 • Cursor 讀取你開的資料夾，對所有檔案做雜湊（hash），再組成 Merkle Tree。
 • .gitignore、.cursorignore 裡列出的檔案／資料夾不會被管。
➤ 每 10 分鐘只傳變動檔案
 • Cursor 把 Merkle Tree 傳到伺服器。
 • 之後每十分鐘比對 hash，只補上變動的檔案，節省頻寬。

⸻

伺服器端怎麼存？
➤ 切片＋向量化後放在 Turbopuffer
 1. 伺服器把檔案拆成小區塊（chunk）並做 embedding（向量）。
 2. 每個向量旁邊會存一段「經過混淆的相對檔案路徑」＋「該 chunk 的行數區間」。
 3. 同時把每個 chunk 的 hash 當 Key，丟到 AWS 快取，方便同一團隊二次索引時秒傳。

⸻

查詢（Inference）怎麼跑？
➤ 本地＋後端一起合作
 1. 用你提問的內容做 embedding。
 2. 在 Turbopuffer 做最近鄰搜尋，挑出相關 chunk 的「混淆路徑＋行數」。
 3. Client 根據路徑／行數，在本機把真正的程式碼片段讀出來，再回傳伺服器，用來回答問題。
 4. 隱私模式：純文字程式碼不會留在伺服器或向量庫，只短暫經過記憶體計算完就丟棄。

⸻

隱私與安全細節
➤ .cursorignore 可阻擋特定檔案：想完全不讓某檔案上雲，就寫在這裡。
➤ 路徑混淆
 • 先把路徑用 /、. 切片，再用存在本機的密鑰＋固定 6 byte nonce 加密每一段。
 • 雖然還看得出大概的資料夾層次，細節已被遮蔽，偶爾會碰到 nonce 衝突。
➤ Embedding 可被反推？
 • 學術上有人能在掌握模型的前提下，把向量還原出部分字串。
 • Cursor 認為在它的使用情境下難度較高，但若有人入侵向量庫，仍可能推敲出一些程式內容。
➤ Git 歷史也會被索引
 • 只存 commit SHA、parent 及「混淆檔名」。
 • 為了讓同一 Repo、同一團隊共用資料結構，混淆用的金鑰是從最近幾個 commit 的內容 hash 推導而來。
 • 不會存 commit message、檔案實體或 diff。

⸻

真實使用上的坑
➤ 伺服器有時忙不過來
 • 高峰期可能上傳失敗，需要重傳。
 • 你若抓網路封包，看 repo42.cursor.sh 可能會覺得流量比預期大，就是因為重傳。

⸻

一句話總結
Cursor 的「Codebase Indexing」就是：在你本機先做差異化掃描，雲端只存向量和經過混淆的路徑，再把真正程式碼片段留在本機計算，如此讓 AI 能懂整個專案，又盡量降低原始碼外流風險。
