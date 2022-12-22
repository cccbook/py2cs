# 來源 -- https://statistics-using-python.blogspot.com/2019/08/one-way-analysis-of-variancesone-way.html
wt = [61.8,65.1,61.7,63.3,78.8,79.5,76.0,73.4,77.3,70.5,72.6,71.7,72.0,71.1,60.3,63.8,64.1,61.4,60.9]
cl = ["F1","F1","F1","F1","F2","F2","F2","F2","F2","F3","F3","F3","F3","F3","F4","F4","F4","F4","F4"]
dat = {'Weight':wt,'Feed':cl}
import pandas as pd
df = pd.DataFrame(dat)
import statsmodels.api as sm
from statsmodels.formula.api import ols
mod = ols('Weight ~ Feed', data = df).fit()
r = sm.stats.anova_lm(mod, typ = 2)
print(r)

'''
結果: ANOVA table如下
            sum_sq    df          F        PR(>F)
Feed      734.8245   3.0  80.123922  1.845486e-09
Residual   45.8555  15.0        NaN           NaN
p = 1.845e-9 < 0.05，不接受H0: μ1 = μ2 = μ3  = μ4，飼料效果不同。
飼料效果不同是誰跟誰有差呢?要進一步做multiple comparison:
'''

from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison
mc = MultiComparison(df['Weight'], df['Feed'])
tkresult = mc.tukeyhsd()
print(tkresult)
'''
結果如下表:
Multiple Comparison of Means - Tukey HSD,FWER=0.05
===============================================
group1 group2 meandiff  lower    upper   reject
-----------------------------------------------
  F1     F2    14.025  10.6443  17.4057   True (有差)
  F1     F3    8.605    5.2243  11.9857   True (有差)
  F1     F4    -0.875  -4.2557   2.5057  False (沒差)
  F2     F3    -5.42   -8.6073  -2.2327   True (有差)
  F2     F4    -14.9   -18.0873 -11.7127  True (有差)
  F3     F4    -9.48   -12.6673 -6.2927   True (有差)
'''
