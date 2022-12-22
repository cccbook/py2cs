## AlphaGo圍棋程式

* https://github.com/junxiaosong/AlphaZero_Gomoku
* https://github.com/PolyKen/15_by_15_AlphaGomoku
* https://github.com/wndeng/Gomoku-AI
* C -- https://github.com/yffbit/gomoku


### AlphaGo 三部曲

| 十分鐘系列的《輕知識》 (PDF)  |  網路版  | 加長版 | 教學錄影 | 
|--------|-----------|----|-----|
|  用十分鐘瞭解《電腦到底是怎麼下棋的》  | [SlideShare](http://www.slideshare.net/ccckmit/ss-59361780)   | | |
|  用十分鐘瞭解 《AlphaGo的幾個可能弱點》  | [SlideShare](http://www.slideshare.net/ccckmit/alphago-59482042)   | | |
|  AlphaGo in Depth (Mark Chang)  | [SlideShare](http://www.slideshare.net/ckmarkohchang/alphago-in-depth)   | | |
|  [用30分鐘深入瞭解《AlphaGo圍棋程式的設計原理》](../slide/30minAlphaGo3in1.pdf)  | [SlideShare](http://www.slideshare.net/ccckmit/30alphago)   | |  | [Facebook](https://www.facebook.com/ccckmit/posts/10153908393201893) | 

### 內容

策略網路減少搜尋分支數，值網路減少搜尋深度。意思是下棋當中還是有用《對局樹搜尋》，但到底對局當中是用 MinMax 搜尋還是用《蒙地卡羅搜尋》呢？

### 參考文獻

* [AlphaGo推手黃士傑現身分享奪勝祕訣 │20160318 中視新聞LIVE直播](https://www.youtube.com/watch?v=1fnlIhRSIYU)
 * 策略網路減少搜尋分支數，值網路減少搜尋深度。意思是下棋當中還是有用《對局樹搜尋》，但到底對局當中是用 MinMax 搜尋還是用《蒙地卡羅搜尋》呢？
* [AlphaGo in Depth](http://www.slideshare.net/ckmarkohchang/alphago-in-depth) , Mark Chang, 超強，讚讚讚！
 * <http://cpmarkchang.logdown.com/>
* [用十分鐘瞭解 《AlphaGo的幾個可能弱點》](http://www.slideshare.net/ccckmit/alphago-59482042)
 * [Introduction to Monte Carlo Tree Search](https://jeffbradberry.com/posts/2015/09/intro-to-monte-carlo-tree-search/)
 * [淺談Alpha Go所涉及的深度學習技術](https://dotblogs.com.tw/allanyiin/2016/03/12/222215)
* [一张图解AlphaGo原理及弱点](http://mp.weixin.qq.com/s?__biz=MzIxNjE3MTM5OA%3D%3D&mid=402241411&idx=1&sn=98557fdc359a17af9ab6b1ed7e09854a&scene=2&srcid=0314rM6ivyxIaEMfKIaW167Z&from=timeline&isappinstalled=0#wechat_redirect)

### 蒙地卡羅搜尋

* [蒙地卡羅樹搜尋](https://zh.wikipedia.org/wiki/%E8%92%99%E7%89%B9%E5%8D%A1%E6%B4%9B%E6%A0%91%E6%90%9C%E7%B4%A2)
* [Introduction to Monte Carlo Tree Search](https://jeffbradberry.com/posts/2015/09/intro-to-monte-carlo-tree-search/)
 * [Bandit based Monte-Carlo Planning](https://www.lri.fr/~sebag/Examens_2008/UCT_ecml06.pdf)
* [陳俊嶧碩士論文:一個蒙地卡羅之電腦圍棋程式之設計](https://ir.nctu.edu.tw/bitstream/11536/45925/1/558001.pdf)
* 專案 -- https://github.com/dsesclei/mcts/tree/master/js/ai

