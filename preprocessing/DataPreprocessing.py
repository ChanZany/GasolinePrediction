import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
from DataCleanUtils import data_clean

# todo 读取数据
df = pd.read_csv("../data/sample.csv", index_col=['编号', '时间']).iloc[:, 0:-1]
df = df.loc[df.notnull().all(axis=1)]

# TODO 准备训练数据集
ori_features = np.hstack((df.iloc[:, 0:9].values, df.iloc[:, 10:].values))
# ori_features = df.iloc[:, 1:]
ori_labels = df.iloc[:, 9].values

# todo 清洗数据，应该还要返回保留行的下标，对应的label才能取出来
ori_features, ori_labels = data_clean(ori_features, ori_labels)

# todo 特征归一化
std = StandardScaler()
std.fit(ori_features)
std_features = std.transform(ori_features)

# todo 特征降维
n_components = 20  # 超参数 可调
pca = PCA(n_components=n_components)
pca.fit(std_features)
features = pca.transform(std_features)

# features.shape[1]
# todo 将降维后的特征作为x,对应下标的RON损失作为y
X = np.array(features, dtype=np.float32)
y = np.array(ori_labels, dtype=np.float32)

# todo 将清洗后的数据写入磁盘
pd.DataFrame(np.hstack((X, y.reshape(X.shape[0], -1)))).to_csv("../data/train2_data.csv", header=None, index=None)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)

"""
使用机器学习算法：效果不理想 0.23412555147403824
"""

# TODO 进行训练(线性回归)
model = LinearRegression()
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print(score)
for i in range(len(y_test)):
    print(
        "predict:", model.predict(X_test[i].reshape(1, -1 ))[0],
        "\ttarget:", y_test[i]
    )
