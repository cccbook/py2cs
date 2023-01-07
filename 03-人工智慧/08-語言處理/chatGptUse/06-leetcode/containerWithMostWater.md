# 

* https://leetcode.com/problems/container-with-most-water/

Write a python code to solve the following problem

You are given an integer array height of length n. There are n vertical lines drawn such that the two endpoints of the ith line are (i, 0) and (i, height[i]).

Find two lines that together with the x-axis form a container, such that the container contains the most water.

Return the maximum amount of water a container can store.

以下是一個 python 程式，用來解決上述問題：

```py
def maxArea(height):
    # 初始化最大面積為 0
    max_area = 0

    # 從左到右枚舉每一根立柱
    for i in range(len(height)):
        # 從右到左枚舉每一根立柱
        for j in range(i + 1, len(height)):
            # 計算當前組合的面積
            area = min(height[i], height[j]) * (j - i)
            # 更新最大面積
            max_area = max(max_area, area)

    # 返回最大面積
    return max_area

```

程式流程如下：

1. 初始化最大面積為 0。
2. 從左到右枚舉每一根立柱，並從右到左枚舉每一根立柱。
3. 計算當前組合的面積。
4. 更新最大面積。
5. 返回最大面積。

ccc: 貼上去之後，可以正確執行，結果也正確，但 submit後 leetcode 說 Time Limit Exceeded

47 / 60 testcases passed