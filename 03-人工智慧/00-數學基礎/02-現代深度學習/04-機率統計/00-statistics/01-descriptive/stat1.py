import statistics
import scipy

x = [58,57,54,59,53,56,58,59,60]

def stat(x):
    print('x=', x)
    print('平均值:mean(x)=', statistics.mean(x))
    print('中位數:median(x)=', statistics.median(x))
    print('眾數:mode(x)=', statistics.mode(x))
    print('幾何平均數:gmean(x)=', scipy.stats.gmean(x))
    print('調和平均數:hmean(x)=', scipy.stats.hmean(x))
    print('變異數:variance(x)=', statistics.variance(x))
    print('標準差:stdev(x)=', statistics.stdev(x))
    print('變異係數:variation(x)=', scipy.stats.variation(x))
    print('=', )
    print('=', )
    print('=', )
    print('=', )


stat(x)
