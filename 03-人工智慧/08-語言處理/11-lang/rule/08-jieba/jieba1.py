# 斷詞示例

import jieba

seg_list = jieba.cut("我来到北京清华大学", cut_all=False)
print("Default Mode: " + "/ ".join(seg_list))  # 精确模式

#輸出 Default Mode: 我/ 来到/ 北京/ 清华大学
