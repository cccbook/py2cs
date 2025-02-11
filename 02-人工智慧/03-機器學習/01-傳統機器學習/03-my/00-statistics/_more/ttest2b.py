from scipy import stats
np.random.seed(12345678)
# Test with sample with identical means:

rvs1 = stats.norm.rvs(loc=5,scale=10,size=500)
rvs2 = stats.norm.rvs(loc=5,scale=10,size=500)
print(stats.ttest_ind(rvs1,rvs2))

# (0.26833823296239279, 0.78849443369564776)
print(stats.ttest_ind(rvs1,rvs2, equal_var = False))
# (0.26833823296239279, 0.78849452749500748)
# ttest_ind underestimates p for unequal variances:

rvs3 = stats.norm.rvs(loc=5, scale=20, size=500)
print(stats.ttest_ind(rvs1, rvs3))
# (-0.46580283298287162, 0.64145827413436174)
print(stats.ttest_ind(rvs1, rvs3, equal_var = False))
# (-0.46580283298287162, 0.64149646246569292)
# When n1 != n2, the equal variance t-statistic is no longer equal to the unequal variance t-statistic:


rvs4 = stats.norm.rvs(loc=5, scale=20, size=100)
printstats.ttest_ind(rvs1, rvs4)
(-0.99882539442782481, 0.3182832709103896)
stats.ttest_ind(rvs1, rvs4, equal_var = False)
(-0.69712570584654099, 0.48716927725402048)
T-test with different means, variance, and n:


rvs5 = stats.norm.rvs(loc=8, scale=20, size=100)
stats.ttest_ind(rvs1, rvs5)
(-1.4679669854490653, 0.14263895620529152)
stats.ttest_ind(rvs1, rvs5, equal_var = False)
(-0.94365973617132992, 0.34744170334794122)