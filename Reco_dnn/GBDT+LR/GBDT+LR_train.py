import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import log_loss

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder

def data_pre(path):
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
    continuous_feature = ['I'+str(i+1) for i in range(13)]
    category_feature = ['C'+str(i+1) for i in range(26)]

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
    return train, target, test, x_train, x_val, y_train, y_val

# GBDT负责对各个特征进行交叉和组合， 把原始特征向量转换为新的离散型特征向量， 然后在使用逻辑回归模型
def gbdt_lr_model(train, target, test, x_train, x_val, y_train, y_val):
    
    # 基于Scikit-learn API
    # lgb.LGBMClassifier默认的boosting_type为(default="gbdt")
    gbm = lgb.LGBMClassifier(objective='binary', # ‘binary’ or ‘multiclass’ for LGBMClassifier
                        subsample= 0.8, #Subsample ratio of the training instance.
                        min_child_weight= 0.5, # Minimum sum of instance weight(hessian) needed in a child(leaf).
                        colsample_bytree= 0.7, # Subsample ratio of columns when constructing each tree.
                        num_leaves=100, # Maximum tree leaves for base learners.
                        max_depth = 12, # Maximum tree depth for base learners, -1 means no limit.
                        learning_rate=0.01, # Boosting learning rate.
                        n_estimators=1000, # Number of boosted trees to fit.
                        )

    gbm.fit(x_train, y_train,
            eval_set = [(x_train, y_train), (x_val, y_val)], # A list of (X, y) tuple pairs to use as a validation sets for early-stopping.
            eval_names = ['train', 'val'], # Names of eval_set.
            eval_metric = 'binary_logloss',
            early_stopping_rounds = 100,
            )
    
    model = gbm.booster_ # The underlying Booster of this model.

    # 拿训练好的GBDT模型去交叉特征
    gbdt_feats_train = model.predict(train, pred_leaf=True)
    gbdt_feats_test = model.predict(test, pred_leaf=True)
    gbdt_feats_name = ['gbdt_leaf_' + str(i) for i in range(gbdt_feats_train.shape[1])]
    df_train_gbdt_feats = pd.DataFrame(gbdt_feats_train, columns = gbdt_feats_name) 
    df_test_gbdt_feats = pd.DataFrame(gbdt_feats_test, columns = gbdt_feats_name)
    # 组合新的特征去交给lr训练
    new_train = pd.concat([train, df_train_gbdt_feats], axis = 1)
    new_test = pd.concat([test, df_test_gbdt_feats], axis = 1)
    new_train_len = new_train.shape[0]
    new_data = pd.concat([new_train, new_test])

    continuous_feature = ['I'+str(i+1) for i in range(13)]
    category_feature = ['C'+str(i+1) for i in range(26)]

    # # 连续特征归一化
    scaler = MinMaxScaler()
    for col in continuous_feature:
        new_data[col] = scaler.fit_transform(new_data[col].values.reshape(-1, 1))

    for col in gbdt_feats_name:
        new_onehot_feats = pd.get_dummies(new_data[col], prefix = col)
        new_data.drop([col], axis = 1, inplace = True)
        new_data = pd.concat([new_data, new_onehot_feats], axis = 1)

    new_train = new_data[: new_train_len]
    new_test = new_data[new_train_len:]

    new_x_train, new_x_val, new_y_train, new_y_val = train_test_split(new_train, target, test_size = 0.3, random_state = 2018)

    lr = LogisticRegression()
    lr.fit(new_x_train, new_y_train)
    tr_logloss = log_loss(new_y_train, lr.predict_proba(new_x_train)[:, 1])
    print('tr-logloss: ', tr_logloss)
    val_logloss = log_loss(new_y_val, lr.predict_proba(new_x_val)[:, 1])
    print('val-logloss: ', val_logloss)
    y_pred = lr.predict_proba(new_test)[:, 1] # Return the predicted probability for each class for each sample.
    print(y_pred[:10])

if __name__ == "__main__":
    train, target, test, x_train, x_val, y_train, y_val = data_pre(path='data/')
    gbdt_lr_model(train, target, test, x_train, x_val, y_train, y_val)
