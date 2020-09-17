import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import numpy as np



## 读取数据并清理数据
df = pd.read_csv("../data/sample.csv",index_col='编号').iloc[:,0:-1]
df = df.loc[df.notnull().all(axis=1)]


# TODO 准备训练数据集
# todo 特征降维
ori_features =df.iloc[:,15:].to_numpy()
ori_labels = df.iloc[:,10].to_numpy()

# TODO 将数据集写入csv文件

# todo 特征归一化
std = StandardScaler()
std.fit(ori_features)
std_features = std.transform(ori_features)

n_components = 10 # 超参数 可调
pca = PCA(n_components=n_components)
pca.fit(std_features)
features = pca.transform(std_features)

# features.shape[1]
#将降维后的特征作为x,对应下标的RON损失作为y
X = np.array(features,dtype=np.float32)
y = np.array(ori_labels,dtype=np.float32)
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=666)

"""
使用机器学习算法：效果不理想 0.17514387997452596
"""
# TODO 进行训练(线性回归)
model = Pipeline([
    ("pca",PCA(n_components=10)),
    ("linerReg",LinearRegression())
])
model.fit(X_train,y_train)
score = model.score(X_test,y_test)
print(score)
























