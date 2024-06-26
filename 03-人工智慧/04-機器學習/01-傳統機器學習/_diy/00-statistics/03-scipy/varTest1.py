# 來源 -- https://statistics-using-python.blogspot.com/2019/08/one-way-analysis-of-variancesone-way.html
# 7. 檢查資料是否為相同變異數 (H0: s12 = s22 = s32  = s42):
# 方法: Levene test for equal variances (parametric test)
dat1 = [61.8,65.1,61.7,63.3]
dat2 = [78.8,79.5,76.0,73.4,77.3]
dat3 = [70.5,72.6,71.7,72.0,71.1]
dat4 = [60.3,63.8,64.1,61.4,60.9]
import scipy.stats
print(scipy.stats.levene(dat1, dat2, dat3, dat4, center = 'mean'))
# 結果: LeveneResult(statistic=1.9584062218957583, pvalue=0.16363937651728697)
# p = 0.1636 > 0.05，接受H0: s12 = s22 = s32  = s42。
# 相同變異數表示樣本來自相同母體(population)，不同變異數表示樣本取樣自不同母體。