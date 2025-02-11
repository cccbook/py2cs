## Baseball 棒球問答系統 – 理解程度較深

Baseball 是一個專門用來回答美國棒球紀錄之問題的交談系統18，包含一個有關美國棒球運動比賽紀錄的資料庫，並根據這個資料庫所記載的資料回答使用者的問題，其方法為格位填充法，由於問答內容
限定在一個資料庫中，因此、使用的格位只要包含這些欄位即可。

> Baseball: An Automatic Question Answerer, B. Green, A. Wolf, C. Chomsky, and K. Laughery. Computers and Thought, Massachusetts: AAAI Press, (1963)

以下是其資料庫的一些記錄，我們用這個表格來說明此系統的運作過程。

Place | Month | Day | Game | Winner/Score | Loser/Score
------|-------|-----|------|--------------|---------------
Cleveland | July |  6 |  95 |  White Sox / 2  | Indian / 0
Boston  | July  | 7  | 96  | Red Sox /5  | Yankees / 3
Detroit July |  7  | 97  | Tiger / 10  | Athletics / 2
Boston July  | 15  | 98  | Yankees/7  | Red Sox/4

BASEBALL 系統提出一個稱為規格串列(Specification list) 的資料結構以便進行格位填充，下表是這個系統根據每個問題所建立的規格串列表：

Question | Specification List
---------|----------------------------------
Where did the Red Sox play on July 7 ? | Team = Red Sox Place = ? Month = July Day = 7
What team won 10 games in July ?  | Team(wining) = ? Game(number_of)=10 Month = July

問題是要如何建立規格串列呢？其方法是採用字典查詢，並對每一個字訂定其語意。

以下是一些字及其語意的對應關係表

Word | Semantic
-----|-----------------
Team | Team = (blank)
Red Sox | Team = Red Sox
Who | Team = ?
Winning | Adj : Winning
Boston | Place = Boston Or Team = Red Sox
The | No meaning
Did | No meaning

以下是其主要流程，我們將以 How many games did the Yankee play in July ? 為例，解釋每一個步驟的動作：

1 - Question Read in : How many games did the Yankee play in July ?

2 - Dictionary Look-up

Word | Semantic
-----|------------------------
How many | Adj : Number_of = ?
games | Game = (blank)
Did | No meaning
The | No meaning
Yankee | Team = Yankee
Play | No meaning
In | No meaning
July | Month = July

3 - Syntax (形成短語 phrase，以便建構出 modifier, 例如上述的 winning)

[How many games] did [the Yankees] play (in [July]) ?

[X] : 代表 noun phrase
(Y) : 代表 adverbial phrase 或 prepositional phrase

4 - Content Analysis (根據字典的語意組合正確的規格串列, 例如上述的 Boston 到底是那個意義呢？)

Phrase | Semantic
-------|----------------------------
[How many games] | Game(Number_of) = ?
Did | No meaning
[the Yankees] | Team = Yankee
Play | No meaning
(in July) | Month = July

5 - Retrieve records(從資料庫中取出記錄)

Place    | Month | Day | Game Winner/Score | Loser/Score
---------|-------|------|------------------|------------
Boston   | July |  7  | 96  | Red Sox /5  | Yankees / 3
Boston   | July |  15  | 98  | Yankees/7  | Red Sox/4

6 - Response (直接顯示擷取出的紀錄)

> Game(Number_of) = ? => count(Game) = ?
> 
> 回答： Two Games 

