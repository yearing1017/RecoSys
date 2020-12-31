import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder

def data_pre(path = 'data/'):
    df_train = pd.read_csv(path + 'train.csv')
    df_test = pd.read_csv(path + 'test.csv')

    # 简单的数据预处理
    # 去掉id列，axis=1代表删除列，默认不加为删除行，把测试集和训练集合并， 填充缺失值
    df_train.drop(['Id'], axis=1, inplace=True)
    df_test.drop(['Id'], axis=1, inplace=True)
    df_test['Label'] = -1
    data = pd.concat([df_train, df_test])
    # 将缺失的nan值填充为-1
    data.fillna(-1, inplace=True)

    # 下面把特征列分开处理
    continuous_fea = ['I'+str(i+1) for i in range(13)]
    category_fea = ['C'+str(i+1) for i in range(26)]

    # 离散特征one-hot编码
    for col in category_feature:
        # pd.get_dummies方法可对数据进行onehot编码，prefix为指定新的列名的前缀
        onehot_feats = pd.get_dummies(data[col], prefix = col)
        data.drop([col], axis = 1, inplace = True)
        data = pd.concat([data, onehot_feats], axis = 1)

    # 编码完毕之后，将train和test分开
    train = data[data['Label'] != -1]
    # 直接弹出label这一列
    target = train.pop('Label')
    test = data[data['Label'] == -1]
    test.drop(['Label'], axis = 1, inplace = True)

    # 划分训练和验证数据集
    x_train, x_val, y_train, y_val = train_test_split(train, target, test_size = 0.2, random_state = 2020)
    return train, test, x_train, x_val, y_train, y_val