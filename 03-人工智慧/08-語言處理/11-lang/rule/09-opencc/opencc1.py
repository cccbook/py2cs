from opencc import OpenCC

cc = OpenCC('t2s')
text = '投票當天需攜帶投票通知單、國民身分證及印章，若沒有收到投票通知書，可以向戶籍所在地鄰長查詢投票所，印章則是可以用簽名代替，至於身分證則是一定要攜帶。'

print(cc.convert(text))
