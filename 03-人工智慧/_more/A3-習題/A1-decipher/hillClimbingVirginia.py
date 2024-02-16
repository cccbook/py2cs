from hillClimbing import hillClimbing # 引入爬山演算法類別
from solutionVirginia import SolutionVirginia # 引入平方根解答類別
import virginia

plain = "This is a book. That is a cat. I am a boy. One of my boy go to school today."
key = [5,2,4]
# key = [9,3,7]
etext = virginia.encrypt(plain, key)
print('etext=', etext)
# 執行爬山演算法
s = hillClimbing(SolutionVirginia.init(etext, [0,0,0]), 30000, 1000)
dtext = virginia.decrypt(etext, s.v)
print('dtext=', dtext)
