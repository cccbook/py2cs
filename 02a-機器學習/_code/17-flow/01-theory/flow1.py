import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# 讀取數據集
df = pd.read_csv('data.csv')

# 處理缺失值，使用均值填充
imputer = SimpleImputer(strategy='mean')
df_imputed = imputer.fit_transform(df)

# 特徵標準化
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_imputed)

# 拆分訓練集與測試集
X_train, X_test, y_train, y_test = train_test_split(df_scaled, df['target'], test_size=0.3, random_state=42)
