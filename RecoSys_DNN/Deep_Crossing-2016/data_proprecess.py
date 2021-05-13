import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def data_pre():
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    print(train_df.shape, test_df.shape) # (1599, 41) (400, 40)
    # 合并数据
    label = train_df.pop('Label')
    data_df = pd.concat((train_df,test_df))
    data_df.drop(['Id'], axis=1, inplace=True)
    # 区分类别特征和数值特征
    sparse_feas = [col for col in data_df.columns if col[0] == 'C']
    dense_feas = [col for col in data_df.columns if col[0] == 'I']
    # 填充缺失值，类别类填充为-1，连续型的数值填充为0 
    data_df[sparse_feas] = data_df[sparse_feas].fillna('-1')
    data_df[dense_feas] = data_df[dense_feas].fillna(0)

    # 类别特征编码，此处的编码方法不是onehot类型编码，而是将所有的类别格式化为连续的index，再进行类别的编码
    for feat in sparse_feas:
        le = LabelEncoder()
        data_df[feat] = le.fit_transform(data_df[feat])

    # 数值特征归一化
    mms = MinMaxScaler()
    data_df[dense_feas] = mms.fit_transform(data_df[dense_feas])

    # 完成之后分开测试集和训练集，df取行操作为在df[]中写数字，取列操作为在[]中写字符串
    train = data_df[:train_df.shape[0]]
    test = data_df[train_df.shape[0]:]
    train['Label'] = label

    #划分训练和验证数据集
    train_set, val_set = train_test_split(train, test_size = 0.2, random_state=2020)
    train_set['Label'].value_counts() 
    val_set['Label'].value_counts()
    # 随机划分之后，重置id并删除
    train_set.reset_index(drop=True, inplace=True)
    val_set.reset_index(drop=True, inplace=True)

    train_set.to_csv('preprocessed_data/train_set.csv', index=0)
    val_set.to_csv('preprocessed_data/val_set.csv', index=0)
    test.to_csv('preprocessed_data/test.csv', index=0)

if __name__ == "__main__":
    data_pre()