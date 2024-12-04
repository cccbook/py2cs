import numpy as np  # 引入 NumPy 庫以進行數學運算

# 定義 PCA 類
class PCA:
    def __init__(self, n_components):
        self.n_components = n_components  # 設置主成分的數量
        self.components = None  # 用於儲存主成分向量
        self.mean = None  # 用於儲存數據的均值

    def fit(self, X):
        # 對數據進行均值中心化
        self.mean = np.mean(X, axis=0)  # 計算每個特徵的均值
        X = X - self.mean  # 將均值從原始數據中減去，以進行均值中心化

        # 計算協方差矩陣，注意樣本作為列
        cov = np.cov(X.T)  # 計算標準化後數據的協方差矩陣

        # 計算特徵值和特徵向量
        eigenvectors, eigenvalues = np.linalg.eig(cov)  # 使用 NumPy 計算特徵值和特徵向量

        # 將特徵向量轉置，方便後續計算
        eigenvectors = eigenvectors.T  # 轉置特徵向量矩陣，使每一列為一個特徵向量

        # 根據特徵值對特徵向量進行排序
        idxs = np.argsort(eigenvalues)[::-1]  # 獲取特徵值從大到小的索引
        eigenvalues = eigenvalues[idxs]  # 根據索引重新排列特徵值
        eigenvectors = eigenvectors[idxs]  # 根據索引重新排列特徵向量

        # 選擇前 n_components 個特徵向量作為主成分
        self.components = eigenvectors[:self.n_components]  # 儲存主成分

    def transform(self, X):
        # 將數據投影到主成分上
        X = X - self.mean  # 對新數據進行均值中心化
        return np.dot(X, self.components.T)  # 返回數據在主成分上的投影

# 測試代碼
if __name__ == "__main__":
    # 引入繪圖庫
    import matplotlib.pyplot as plt
    from sklearn import datasets  # 引入 scikit-learn 的數據集

    # 載入 Iris 數據集
    data = datasets.load_iris()  # 可替換成 datasets.load_digits() 來使用數字數據集
    X = data.data  # 特徵數據
    y = data.target  # 標籤數據

    # 將數據投影到前兩個主成分上
    pca = PCA(2)  # 初始化 PCA 對象，設置主成分數量為 2
    pca.fit(X)  # 擬合數據，計算主成分
    X_projected = pca.transform(X)  # 將原始數據轉換到主成分空間

    # 輸出原始數據和轉換後數據的形狀
    print("Shape of X:", X.shape)  # 輸出原始數據的形狀
    print("Shape of transformed X:", X_projected.shape)  # 輸出轉換後數據的形狀

    # 提取主成分的 X 和 Y 值
    x1 = X_projected[:, 0]  # 第一主成分
    x2 = X_projected[:, 1]  # 第二主成分

    # 繪製散點圖
    plt.scatter(
        x1, x2, c=y, edgecolor="none", alpha=0.8, cmap=plt.get_cmap("viridis", 3)
    )
    
    # 標記坐標軸
    plt.xlabel("Principal Component 1")  # X 軸標籤
    plt.ylabel("Principal Component 2")  # Y 軸標籤
    plt.colorbar()  # 添加顏色條以顯示類別
    plt.show()  # 顯示繪製的圖形
