# BERT

* [2021 iThome 鐵人賽:BERT系列Model的應用、訓練技巧與實戰 ](https://ithelp.ithome.com.tw/users/20127672/ironman/4652)

## 1

* https://ithelp.ithome.com.tw/m/articles/10260092

![](https://ithelp.ithome.com.tw/upload/images/20210904/20127672yDEw1XDTE2.jpg)

## 2

* https://ithelp.ithome.com.tw/m/articles/10260396

BERT 裡面還是用高維 vector 表示一個詞 (例如 768 維)，但不是 one-hot

Token Embeddings、Segment Embeddings、Position Embeddings是已經包含在BERT模型中的嵌入，在實際應用過程中，我們只要能撈取到對應的Embeddings即可，那麼怎麼撈取呢？這些嵌入是以字典的形式進行儲存，例如對於Token Embeddings，每一個不同的詞（Token）對應固定ID的嵌入。所以我們只要把詞語轉換為對應的數字id即可。而BERT模型在釋出時也會提供相應的字典，讓你可以進行自動對應。

## 3

* https://ithelp.ithome.com.tw/m/articles/10261228

![](https://ithelp.ithome.com.tw/upload/images/20210905/20127672biWv2mIAdN.jpg)

Token Embeddings：也就是詞嵌入。只是BERT模型的輸入的Token embeddings是最淺層的詞嵌入，只能代表詞語的淺層特徵（例如字符長度之類的），也沒有包含上下文脈絡，因為還沒有經過與序列中其他token的運算。

Segment Embeddings：分段嵌入。只有兩種不同的向量，用於在輸入是兩個句子時分辨 token 是屬於哪個句子。第一句子的每一個詞對應相同的Segment Embedding，第二句對應第二種Segment embedding。

Position Embeddings：表示序列位置的嵌入。因為BERT是同時輸入做平行計算，而非一步步按照序列進行輸入，所以無法自然得知序列的前後順序，需要一個位置嵌入來補足。

對應到

這部分的三個輸入（下方括號內為此輸入在Transformers中的變量名稱）分別有：

tokens_tensor（input_ids）:每個token的索引值，用於對應Token Embeddings。可以用BERT模型提供的vocab.txt查到。

segments_tensor（token_type_ids）:對應Segment Embeddings，識別句子界限，第一句中的每個詞給0，第二句中每個詞給1。

masks_tensor（attention_mask）:排除佔位符（<pad>）用，界定自注意力機制範圍，為1則是有意義的文本序列內容，進入後續運算，<pad>對應0，不進入運算。



a.句子對分類

在這裡，BERT作者們推薦的做法是將輸出的[CLS]的768維向量Embedding拿去當作這兩個句子的語義表示，然後接上一個簡單的線性層作為Output Layer。原因是，在BERT模型的設計中，[CLS]會有一個全連接層與其他所有token相連，微調後可以當作聚合了整句的資訊。

## 4

* https://ithelp.ithome.com.tw/m/articles/10261228

## 5

* https://ithelp.ithome.com.tw/m/articles/10261606

例如雖然Google有釋出多語言BERT，但各國的研究者仍傾向使用自己語言所預訓練出的BERT。在中文領域，這方面做得最好的是 [哈工大/訊飛實驗室](https://github.com/ymcui/Chinese-BERT-wwm) ，他們釋出了一系列中文的BERT模型，我實際使用過，效果頗佳。

工程上的改善最著名的是RoBERTa，這是FB的研究者在BERT基礎上改進、重新預訓練所獲得的效果更好的BERT版本。有人認為RoBERTa才是真正的BERT，Google所開發的只是一個未完成的版本。

# 6 -- Transformer

* https://ithelp.ithome.com.tw/m/articles/10261889

![](https://ithelp.ithome.com.tw/upload/images/20210909/201276722282FsyI09.jpg)

![](https://ithelp.ithome.com.tw/upload/images/20210909/20127672gW8Msdj5jW.png)

一個基本事實：Transformer是Seq2seq模型，而Transformer是BERT模型的基本組成結構，但是BERT卻並不是Seq2seq模型

 其實完整的說法是：Google-BERT由Transformer的Encoder部分組成，沒有使用Decoder部分。

 遠親的另一個預訓練語言模型GPT，就是完全用Transformer的Decoder部分進行預訓練的。所以GPT系列難以進行自然語言理解，卻在生成任務上打遍天下無敵手。

 # 7

 # 8

 * https://ithelp.ithome.com.tw/m/articles/10262644

 1.讓預訓練任務更像下游任務
這個領域不太適合一般研究者來實作，畢竟預訓練一個BERT模型所花費的成本已經不是普通研究者或碩博士學生可以承擔的了。但是，我們可以選擇最新的、更合適的預訓練模型來進行我們的下游任務。對於這些模型資源，如果不善加利用，是非常可惜的。舉例而言，之前簡單介紹過用於生成的BART、用於摘要任務的PEGASUS就是這種思考下的產物。其它還有許多類似的模型，之後我們會專門來介紹，我最近發現的QA預訓練模型Splinter就表現不錯。

2.繼續預訓練
不要停止你的預訓練！拿到一個預訓練好的模型之後，你可以繼續在你的下游任務文本上繼續進行預訓練，也可以在相似的領域文本上進行。將繼續預訓練後的模型拿來進行微調，已經被許多論文證明可以提升最終表現。雖然這很費功夫，也花時間和顯卡，但預訓練仍然是王道！

3.連續微調/多任務微調
沒有顯卡，沒有時間繼續預訓練，怎麼辦？那就微調好了。我們不只要在下游任務上微調，我們可以在一些相似任務、相似領域的資料上進行微調。再將微調後的模型拿去做最終任務的微調。當然，除了按順序進行微調，你也可以選擇進行多任務同時進行的微調。雖然參數比較難調，但具有提升模型效果的能力。

4.Promoting/Prefix-Tuning：讓下游任務更像預訓練任務

有沒有什麼專門為了特定任務而預訓練的模型呢？今天就介紹一個剛發佈不久的Splinter模型，專為QA而生，效果優於RoBERTa、SpanBERT等之前的SOTA，在小樣本的訓練上更是特別優秀。

# 9

* https://ithelp.ithome.com.tw/m/articles/10263059


那麼有沒有什麼專門為了特定任務而預訓練的模型呢？今天就介紹一個剛發佈不久的Splinter模型，專為QA而生，效果優於RoBERTa、SpanBERT等之前的SOTA，在小樣本的訓練上更是特別優秀。

Splinter：span-level pointer
