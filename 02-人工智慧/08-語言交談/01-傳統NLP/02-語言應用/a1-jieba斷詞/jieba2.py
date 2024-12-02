import sys
import jieba

seg_list = jieba.cut(sys.argv[1], cut_all=False)
print("斷詞結果: " + "/ ".join(seg_list))

