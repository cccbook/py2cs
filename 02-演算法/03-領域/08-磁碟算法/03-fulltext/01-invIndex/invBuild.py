import invIndex as i2

docs = [
    'Search engine indexing is the collecting, parsing, and storing of data to facilitate fast and accurate information retrieval. Index design incorporates interdisciplinary concepts from linguistics, cognitive psychology, mathematics, informatics, and computer science. An alternate name for the process, in the context of search engines designed to find web pages on the Internet, is web indexing.',
    'In computer science, an inverted index (also referred to as a postings list, postings file, or inverted file) is a database index storing a mapping from content, such as words or numbers, to its locations in a table, or in a document or a set of documents (named in contrast to a forward index, which maps from documents to content). The purpose of an inverted index is to allow fast full-text searches, at a cost of increased processing when a document is added to the database. The inverted file may be the database file itself, rather than its index. It is the most popular data structure used in document retrieval systems,[1] used on a large scale for example in search engines. Additionally, several significant general-purpose mainframe-based database management systems have used inverted list architectures, including ADABAS, DATACOM/DB, and Model 204.',
    '倒排索引（英語：Inverted index），也常被稱為反向索引、置入檔案或反向檔案，是一種索引方法，被用來儲存在全文搜尋下某個單詞在一個文件或者一組文件中的儲存位置的對映。它是文件檢索系統中最常用的資料結構。',
    '資訊檢索（英語：Information Retrieval）是从信息资源集合获得与信息需求相关的信息资源的活动。搜索可以基于全文或其他基于内容的索引。'
]

invIndex = {}
i2.idxBuild(invIndex, docs)
i2.idxSave(invIndex)