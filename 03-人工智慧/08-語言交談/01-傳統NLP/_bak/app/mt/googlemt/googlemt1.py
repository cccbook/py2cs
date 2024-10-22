from googletrans import Translator
translator = Translator()
'''
print('안녕하세요.=', translator.translate('안녕하세요.').text)
print('안녕하세요.=', translator.translate('안녕하세요.', dest='ja').text)
print('안녕하세요.=', translator.translate('veritas lux mea', src='la').text)
'''

# print('Hello, How are you!=', translator.translate('Hello, How are you!', src='en', dest='zh-TW').text)
english = 'The number of micro operations is minimized without impacting the quality of the generated code much. For example, instead of generating every possible move between every 32 PowerPC registers, we just generate moves to and from a few temporary registers. These registers T0, T1, T2 are typically stored in host registers by using the GCC static register variable extension.'
print('======== 英文 ========')
print(english)
print('======== 中文 ========')
print(translator.translate(english, src='en', dest='zh-TW').text)
