# 第 3 章：Lambda 與 Lazy (延遲求值)

## 沒有用延遲求值 -- 結果當掉

檔案: no_lazy.py

```py
def IF(cond,job_true,job_false):
    if cond:
        return job_true
    else:
        return job_false

print(f'IF(True,"Yes","No")={IF(True,"Yes","No")}')

# 階層 FACTORIAL(n) = n!
def FACTORIAL(n): 
  return IF(n==0, 1, n*FACTORIAL(n-1))

print(f'FACTORIAL(3)={FACTORIAL(3)}')

```


## 使用延遲求職 -- 成功執行

檔案: lazy.py

```py
def IF(cond,job_true,job_false):
    if cond:
        return job_true()
    else:
        return job_false()

# 階層 FACTORIAL(n) = n!
def FACTORIAL(n): 
  return IF(n==0, lambda:1, lambda:n*FACTORIAL(n-1))

print(f'FACTORIAL(3)={FACTORIAL(3)}')
print(f'FACTORIAL(5)={FACTORIAL(5)}')

```