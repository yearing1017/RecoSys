import argparse
import gc
import os
import random
import warnings

import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from utils import Logger, evaluate, gen_sub

warnings.filterwarnings('ignore')

seed = 2020
random.seed(seed)

# 命令行参数
parser = argparse.ArgumentParser(description='lightgbm 排序')
parser.add_argument('--mode', default='valid')
parser.add_argument('--logfile', default='test.log')

args = parser.parse_args()

mode = args.mode
logfile = args.logfile

# 初始化日志
os.makedirs('../user_data/log', exist_ok=True)
log = Logger(f'../user_data/log/{logfile}').logger
log.info(f'lightgbm 排序，mode: {mode}')


def train_model(df_feature, df_query):
    # 现在的训练样本:最初的5w验证集+test  ->  经过召回+特征工程之后 -> label不为空的
    df_train = df_feature[df_feature['label'].notnull()]
    # 现在的测试样本就是之前的test，label设为null的行，但是这些用户也召回新闻了
    df_test = df_feature[df_feature['label'].isnull()]
    

    del df_feature
    gc.collect()

    ycol = 'label'
    # 除去这三个列的其他的列的名字
    feature_names = list(
        filter(
            lambda x: x not in [ycol, 'created_at_datetime', 'click_datetime'],
            df_train.columns))
    feature_names.sort()

    model = lgb.LGBMClassifier(num_leaves=64,
                               max_depth=10,
                               learning_rate=0.05,
                               n_estimators=10000,
                               subsample=0.8,
                               feature_fraction=0.8,
                               reg_alpha=0.5,
                               reg_lambda=0.5,
                               random_state=seed,
                               importance_type='gain',
                               metric=None)

    oof = []
    # 取出这两列的内容
    prediction = df_test[['user_id', 'article_id']]
    # 新加一列pred
    prediction['pred'] = 0
    df_importance_list = []

    # 训练模型
    kfold = GroupKFold(n_splits=5)
    # GroupKFold划分：kfold.spliy(X,Y,groups),X为输入特征，Y为标签，groups为按照其中的相同数字进行划分训练和验证
    # 我们有时候可能不希望某一组或某一类型的样本被切分到训练集和测试集中，而是希望这组数据全部都在训练集中或者全部在测试集中，
    # 此处将每个用户的全部放到训练集中 或 验证集中  不被分开
    for fold_id, (trn_idx, val_idx) in enumerate(
            kfold.split(df_train[feature_names], df_train[ycol],df_train['user_id'])):
        X_train = df_train.iloc[trn_idx][feature_names]
        Y_train = df_train.iloc[trn_idx][ycol]

        X_val = df_train.iloc[val_idx][feature_names]
        Y_val = df_train.iloc[val_idx][ycol]

        log.debug(
            f'\nFold_{fold_id + 1} Training ================================\n'
        )

        lgb_model = model.fit(X_train,
                              Y_train,
                              eval_names=['train', 'valid'],
                              eval_set=[(X_train, Y_train), (X_val, Y_val)],
                              verbose=100,
                              eval_metric='auc',
                              early_stopping_rounds=100)
        # 验证集结果的计算
        # 计算类别1的auc，则要用样本预测为1的概率model.predict_proba(testdata)[:,1]
        pred_val = lgb_model.predict_proba(
            X_val, num_iteration=lgb_model.best_iteration_)[:, 1]
        # 拷贝的验证集的三列
        df_oof = df_train.iloc[val_idx][['user_id', 'article_id', ycol]].copy()
        df_oof['pred'] = pred_val # 填入该样本被预测为1的概率
        oof.append(df_oof) # 总的oof包含k折验证的所有验证集的结果

        # 测试集结果的计算
        pred_test = lgb_model.predict_proba(
            df_test[feature_names], num_iteration=lgb_model.best_iteration_)[:,1]
        # 将5次测试的结果加和除5 填入prediction
        prediction['pred'] += pred_test / 5

        df_importance = pd.DataFrame({
            'feature_name':
            feature_names,
            'importance':
            lgb_model.feature_importances_,
        })
        df_importance_list.append(df_importance)
        # 保存模型
        joblib.dump(model, f'../user_data/model/lgb{fold_id}.pkl')
    # k折训练结束后
    # 特征重要性
    df_importance = pd.concat(df_importance_list)
    df_importance = df_importance.groupby([
        'feature_name'
    ])['importance'].agg('mean').sort_values(ascending=False).reset_index()
    log.debug(f'importance: {df_importance}')

    # 生成线下，df_oof为df_train所有的投入到训练的用户的排序结果
    df_oof = pd.concat(oof)
    # user_id升序  pred为1的概率降序排列 越大的越靠前
    df_oof.sort_values(['user_id', 'pred'],
                       inplace=True,
                       ascending=[True, False])
    log.debug(f'df_oof.head: {df_oof.head()}')

    # 线下计算验证集的相关指标
    # 除了test的验证集的人数，线下验证即df_train中所有投入到训练的用户的人数
    total = df_query[df_query['click_article_id'] != -1].user_id.nunique()
    hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50 = evaluate(
        df_oof, total)
    log.debug(
        f'{hitrate_5}, {mrr_5}, {hitrate_10}, {mrr_10}, {hitrate_20}, {mrr_20}, {hitrate_40}, {mrr_40}, {hitrate_50}, {mrr_50}'
    )
    
    # 生成提交文件 prediction是针对最初的test的，即在query中被设为点击文章为-1，label为空的用户
    # 给测试集中每个用户推荐召回结果中排序最高的5条文章
    df_sub = gen_sub(prediction)
    df_sub.sort_values(['user_id'], inplace=True)
    os.makedirs('../prediction_result', exist_ok=True)
    df_sub.to_csv(f'../prediction_result/result.csv', index=False)


def online_predict(df_test):
    # 这时线上，无验证集，此时的df_feature就是df_test,需要推荐的test用户
    #df_test = df_feature[df_feature['label'].isnull()]
    ycol = 'label'
    feature_names = list(
        filter(
            lambda x: x not in [ycol, 'created_at_datetime', 'click_datetime'],
            df_test.columns))
    feature_names.sort()

    prediction = df_test[['user_id', 'article_id']]
    prediction['pred'] = 0

    for fold_id in tqdm(range(5)):
        model = joblib.load(f'../user_data/model/lgb{fold_id}.pkl')
        pred_test = model.predict_proba(df_test[feature_names])[:, 1]
        prediction['pred'] += pred_test / 5

    # 生成提交文件
    df_sub = gen_sub(prediction)
    df_sub.sort_values(['user_id'], inplace=True)
    os.makedirs('../prediction_result', exist_ok=True)
    df_sub.to_csv(f'../prediction_result/result.csv', index=False)


if __name__ == '__main__':
    if mode == 'valid':
        # 特征工程文件包括之前的合并召回结果 + 新建立的特征
        df_feature = pd.read_pickle('../user_data/data/offline/feature.pkl')
        df_query = pd.read_pickle('../user_data/data/offline/query.pkl')
        # 类别型特征
        for f in df_feature.select_dtypes('object').columns:
            lbl = LabelEncoder() # 将n个类别编码为0~n-1之间的整数（包含0和n-1）
            df_feature[f] = lbl.fit_transform(df_feature[f].astype(str))

        train_model(df_feature, df_query)
    else:
        df_feature = pd.read_pickle('../user_data/data/online/feature.pkl')
        online_predict(df_feature)
