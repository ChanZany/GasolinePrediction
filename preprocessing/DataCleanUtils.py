import numpy as np
import pandas as pd


# 使用肖维勒方法（等置信概率）剔除异常值
def clean_bad(column):
    ave = np.mean(column)
    u = np.std(column)
    for i in range(0, len(column)):
        if (abs(column[i] - ave) > 3 * u):
            column[i] = None
            print(i)
        else:
            continue


df = pd.read_csv("../data/sample.csv", index_col=['编号', '时间']).iloc[:, 0:-1]
df = df.loc[df.notnull().all(axis=1)]
ori_features = pd.DataFrame(np.hstack((df.iloc[:, 0:9].values, df.iloc[:, 10:].values)))
print(ori_features.shape)
for i in range(0, len(ori_features)):
    # print(ori_features[0]==0)
    clean_bad(ori_features[i])
    # print(ori_features[0]==0)

print(ori_features.shape)

ori_features = ori_features.dropna()
print(ori_features.shape)
