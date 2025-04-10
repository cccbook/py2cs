# 程式來源 -- https://medium.com/jameslearningnote/%E8%B3%87%E6%96%99%E5%88%86%E6%9E%90-%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-%E7%AC%AC3-5%E8%AC%9B-%E6%B1%BA%E7%AD%96%E6%A8%B9-decision-tree-%E4%BB%A5%E5%8F%8A%E9%9A%A8%E6%A9%9F%E6%A3%AE%E6%9E%97-random-forest-%E4%BB%8B%E7%B4%B9-7079b0ddfbda
'''
Created on July 7, 2019
@author: Terry
@email：terryluohello@qq.com
'''
print(__doc__)

import numpy as np 
import matplotlib.pyplot as plt 

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Parameter
n_classes = 3
plot_colors = "ryb"
plot_step = 0.02

# Load data
iris = load_iris()

for pairidx, pair in enumerate([[0,1],[0,2],[0,3],
                                [1,2],[1,3],[2,3]]):
    # We only take two corresponding features
    X = iris.data[:,pair]
    y = iris.target

    # Train
    clf = DecisionTreeClassifier().fit(X,y)

    # Plot the descision boundary
    plt.subplot(2,3,pairidx + 1)
    x_min, x_max = X[:,0].min() - 1, X[:,0].max() - 1
    y_min, y_max = X[:,1].min() - 1, X[:,1].max() - 1
    xx, yy = np.meshgrid(np.arange(x_min,x_max,plot_step),
                        np.arange(y_min,y_max,plot_step))
    plt.tight_layout(h_pad=0.5,w_pad=0.5,pad=2.5)

    Z = clf.predict(np.c_[xx.ravel(),yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)
    plt.xlabel(iris.feature_names[pair[0]])
    plt.ylabel(iris.feature_names[pair[1]])
    
    # Plot the training points
    for i, color in zip(range(n_classes),plot_colors):
        idx = np.where(y == i)
        plt.scatter(X[idx,0],X[idx,1],c=color,label=iris.target_names[i],
                    cmap=plt.cm.RdYlBu,edgecolors='black',s=15)
plt.suptitle("Decision surface of a decision tree using paired features")
plt.legend(loc='lower right', borderpad=0, handletextpad=0)
plt.axis("tight")

plt.figure()
clf = DecisionTreeClassifier().fit(iris.data, iris.target)
plot_tree(clf, filled=True)
plt.show()