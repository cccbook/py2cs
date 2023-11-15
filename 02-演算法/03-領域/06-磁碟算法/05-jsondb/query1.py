from jsondb import JsonDB

jdb = JsonDB()
jdb.open()
r = jdb.match('unix').match('檔案').where(lambda x:x['id']<50)
print("\njdb.match('unix').match('檔案').where(lambda x:x['id']<50):\n", r)
r = jdb.match('愛因斯坦').where(lambda x:x['id']<100).match('納粹')
print("\njdb.match('愛因斯坦').where(lambda x:x['id']<100).match('納粹'):\n", r)
r = jdb.match('愛因斯坦').where(lambda x:x['id']<100).sort('id', 'DESC')
print("\njdb.match('愛因斯坦').where(lambda x:x['id']<100).sort('id', 'DESC'):\n", r)
r = jdb.match('"id":3', follow=",}")
print("\njdb.match('\"id\":3', follow=\",}\")", r)
r = jdb.select('id', 3)
print("\njdb.select('id', 3)", r)
