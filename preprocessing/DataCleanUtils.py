import numpy as np
import pandas as pd


# 缺失值补全，补全依据是缺失的值认为与它上一行的数据一样,即每个时间段测一次
def fill_nan(demo_df):
    for indexs in demo_df.index:
        for i in range(len(demo_df.loc[indexs].values)):
            if (demo_df.loc[indexs].values[i] == 0):
                # print(indexs, i)
                # print(demo_df.loc[indexs].values[i])
                # todo 取该列的平均值（可修改）
                demo_df.loc[indexs].values[i] = np.mean(demo_df.iloc[:, i])
                # print(demo_df.loc[indexs].values[i])


# 使用肖维勒方法（等置信概率）剔除异常值
def clean_bad(column):
    ave = np.mean(column)
    u = np.std(column)
    for i in range(0, len(column)):
        if (abs(column[i] - ave) > 3 * u):
            column[i] = None
        else:
            continue


def data_clean(data,target):
    clean_data = pd.DataFrame(data)
    fill_nan(clean_data)
    for i in range(0, len(clean_data)):
        clean_bad(clean_data[i])

    cols = ~clean_data.isna().any(axis=1)
    return clean_data[cols].values, target[cols]


# 测试
# df = pd.read_csv("../data/sample.csv", index_col=['编号', '时间']).iloc[:, 0:-1]
# df = df.loc[df.notnull().all(axis=1)]
# ori_features = np.hstack((df.iloc[:, 0:9].values, df.iloc[:, 10:].values))
# ori_labels = df.iloc[:, 9].values
# print(ori_features.shape)
# clean_data,clean_target = data_clean(ori_features,ori_labels)
# print(clean_data.shape)
# print(clean_target.shape)
