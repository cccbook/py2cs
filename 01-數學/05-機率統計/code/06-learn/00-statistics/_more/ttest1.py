import numpy as np
import scipy.stats as st
a = np.array([3.25, 3.27, 3.24, 3.26, 3.24])
mu = a.mean(); s = a.std(ddof=1)
mu0 = 3.25
n = len(a)
t = (mu-mu0)/(s/np.sqrt(n))
ta = st.t.ppf(0.995, n-1)
print("n=", n, "t=", t, "ta=", ta)
