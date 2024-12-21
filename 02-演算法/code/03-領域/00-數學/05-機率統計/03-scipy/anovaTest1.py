# 來源 -- https://statistics-using-python.blogspot.com/2019/08/one-way-analysis-of-variancesone-way.html
# 8. 使用Python計算多組相同變異數獨立樣本ANOVA檢定:
# 方法一: 使用SciPy (scipy.stats.f_oneway)
dat1 = [61.8,65.1,61.7,63.3]
dat2 = [78.8,79.5,76.0,73.4,77.3]
dat3 = [70.5,72.6,71.7,72.0,71.1]
dat4 = [60.3,63.8,64.1,61.4,60.9]
import scipy.stats
scipy.stats.f_oneway(dat1, dat2, dat3, dat4)
# 結果: F_onewayResult(statistic=80.12392188505216, pvalue=1.8454859967040819e-09)
p = 1.845e-9 < 0.05，不接受H0: μ1 = μ2 = μ3  = μ4，飼料效果不同。