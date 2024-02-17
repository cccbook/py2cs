import googletrans
from pprint import pprint

translator = googletrans.Translator()

src = 'Just do it. You can make it.'
print('原文:', src)
print('中文:', translator.translate(src, src="en", dest="zh-tw").text)
print('日文:', translator.translate(src, src="en", dest="ja").text)
print('法文:', translator.translate(src, src="en", dest="fr").text)
print('德文:', translator.translate(src, src="en", dest="de").text)
print('韓文:', translator.translate(src, src="en", dest="ko").text)
print('義大利文:', translator.translate(src, src="en", dest="it").text)
