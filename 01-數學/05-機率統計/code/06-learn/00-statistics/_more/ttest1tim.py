import numpy as np
import scipy.stats as st
a = np.array([0.2,0.4,0.3,0.5,1.1,1,1.2,1.3,0.9,1.1]) # 十個樣本
mu = a.mean(); s = a.std(ddof=1) # mu=樣本平均值 s=標準差
mu0 = 0.25 # 檢定的虛無假設 null hypotheses = 0.25 mg/mL
n = len(a) # 樣本數
t = (mu-mu0)/(s/np.sqrt(n)) # t 值
ta = st.t.ppf(0.9995, n-1) # 雙尾檢定，(1-0.001/2) = 0.9995
print("n=", n, "t=", t, "ta=", ta) # 結果 n= 10 t= 4.260 ta= 4.781
