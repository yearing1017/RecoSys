import argparse
import os
import pickle
import random
import signal
import warnings
from collections import defaultdict
from itertools import permutations
from random import shuffle

import multitasking
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import Logger, evaluate

warnings.filterwarnings('ignore')

max_threads = multitasking.config['CPU_CORES']
multitasking.set_max_threads(max_threads)
multitasking.set_engine('process')
signal.signal(signal.SIGINT, multitasking.killall)

random.seed(2020)

# 命令行参数
parser = argparse.ArgumentParser(description='召回合并')
parser.add_argument('--mode', default='valid')
parser.add_argument('--logfile', default='test.log')

args = parser.parse_args()

mode = args.mode
logfile = args.logfile

# 初始化日志
os.makedirs('user_data/log', exist_ok=True)
log = Logger(f'user_data/log/{logfile}').logger
log.info(f'召回合并: {mode}')


def mms(df):
    user_score_max = {}
    user_score_min = {}

    # 获取用户下的相似度的最大值和最小值
    for user_id, g in df[['user_id', 'sim_score']].groupby('user_id'):
        scores = g['sim_score'].values.tolist()
        # 之前召回时已按相似度大小排好序  
        user_score_max[user_id] = scores[0]
        user_score_min[user_id] = scores[-1]

    ans = []
    # 每个用户召回了n个文章，对每个用户的n个召回结果的相似度进行最大最小值归一化 下面是公式
    for user_id, sim_score in tqdm(df[['user_id', 'sim_score']].values):
        ans.append((sim_score - user_score_min[user_id]) /
                   (user_score_max[user_id] - user_score_min[user_id]) +
                   10**-3)
    return ans

# 多路召回之间肯定存在重复召回的情况，这也是为什么召回策略讲究差异性，其实就是为了减少重复召回的数量。
# 利用重复召回的新闻数量占该路召回数量之比做相似度参考
def recall_result_sim(df1_, df2_):
    df1 = df1_.copy()
    df2 = df2_.copy()

    user_item_ = df1.groupby('user_id')['article_id'].agg(set).reset_index()
    user_item_dict1 = dict(zip(user_item_['user_id'],
                               user_item_['article_id']))

    user_item_ = df2.groupby('user_id')['article_id'].agg(set).reset_index()
    user_item_dict2 = dict(zip(user_item_['user_id'],
                               user_item_['article_id']))

    cnt = 0
    hit_cnt = 0

    for user in user_item_dict1.keys():
        item_set1 = user_item_dict1[user]

        cnt += len(item_set1)

        if user in user_item_dict2:
            item_set2 = user_item_dict2[user]

            inters = item_set1 & item_set2
            hit_cnt += len(inters)

    return hit_cnt / cnt


if __name__ == '__main__':
    if mode == 'valid':
        df_click = pd.read_pickle('user_data/data/offline/click.pkl')
        df_query = pd.read_pickle('user_data/data/offline/query.pkl')

        recall_path = 'user_data/data/offline'
    else:
        df_click = pd.read_pickle('user_data/data/online/click.pkl')
        df_query = pd.read_pickle('user_data/data/online/query.pkl')

        recall_path = 'user_data/data/online'

    log.debug(f'max_threads {max_threads}')

    #recall_methods = ['itemcf', 'w2v', 'binetwork']
    recall_methods = ['itemcf', 'w2v']

    #weights = {'itemcf': 1, 'binetwork': 1, 'w2v': 0.1}
    weights = {'itemcf': 1, 'w2v': 0.1}
    recall_list = []
    recall_dict = {}
    for recall_method in recall_methods:
        recall_result = pd.read_pickle(
            f'{recall_path}/recall_{recall_method}.pkl')
        weight = weights[recall_method]
        # 重复被召回的新闻在多个召回方式中的得分是不一样的，得分是排序时的强特，需要进行得分合并。
        # 对各个召回结果，以用户为单位进行相似度的最大最小归一化。
        recall_result['sim_score'] = mms(recall_result)
        # 召回方法对应其权重
        recall_result['sim_score'] = recall_result['sim_score'] * weight

        recall_list.append(recall_result)
        recall_dict[recall_method] = recall_result

    # 求相似度
    for recall_method1, recall_method2 in permutations(recall_methods, 2):
        score = recall_result_sim(recall_dict[recall_method1],
                                  recall_dict[recall_method2])
        log.debug(f'召回相似度 {recall_method1}-{recall_method2}: {score}')

    # 合并召回结果
    recall_final = pd.concat(recall_list, sort=False)
    # 分数合并测试了sum，mean和max，效果对比见下表。max丢失的消息较多，mean对重复次数多的新闻不公平。
    recall_score = recall_final[['user_id', 'article_id',
                                 'sim_score']].groupby([
                                     'user_id', 'article_id'
                                 ])['sim_score'].sum().reset_index()
    # 去重 用户id和新闻id都相同的召回结果删除掉
    recall_final = recall_final[['user_id', 'article_id', 'label'
                                 ]].drop_duplicates(['user_id', 'article_id'])
    # 联合了多个召回结果的评分
    recall_final = recall_final.merge(recall_score, how='left')

    recall_final.sort_values(['user_id', 'sim_score'],
                             inplace=True,
                             ascending=[True, False])

    log.debug(f'recall_final.shape: {recall_final.shape}')
    log.debug(f'recall_final: {recall_final.head()}')

    # 删除没有召回到真实点击的验证集用户，减少了无用负样本的数量，也提高了后续排序模型的效果，
    # 但是在计算线下指标的时候这部分用户的数量要包括进去，否则指标会虚高。
    # 删除无正样本的训练集用户
    gg = recall_final.groupby(['user_id'])
    useful_recall = []

    for user_id, g in tqdm(gg):
        if g['label'].isnull().sum() > 0:
            useful_recall.append(g)
        else:
            label_sum = g['label'].sum()
            # 因为当时就是按照最后点击构建的，所以一个用户只有一个1
            if label_sum > 1:
                print('error', user_id)
            elif label_sum == 1:
                useful_recall.append(g)

    df_useful_recall = pd.concat(useful_recall, sort=False)
    log.debug(f'df_useful_recall: {df_useful_recall.head()}')

    df_useful_recall = df_useful_recall.sort_values(
        ['user_id', 'sim_score'], ascending=[True,
                                             False]).reset_index(drop=True)

    # 计算相关指标
    if mode == 'valid':
        total = df_query[df_query['click_article_id'] != -1].user_id.nunique()

        hitrate_5_1, mrr_5_1, hitrate_10_1, mrr_10_1, hitrate_20_1, mrr_20_1, hitrate_40_1, mrr_40_1, hitrate_50_1, mrr_50_1 = evaluate(
            recall_final[recall_final['label'].notnull()], total)
        
        log.debug(
            f'召回合并后,未删除无用用户指标: {hitrate_5_1}, {mrr_5_1}, {hitrate_10_1}, {mrr_10_1}, {hitrate_20_1}, {mrr_20_1}, {hitrate_40_1}, {mrr_40_1}, {hitrate_50_1}, {mrr_50_1}'
        )

        hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50 = evaluate(
            df_useful_recall[df_useful_recall['label'].notnull()], total)
        
        log.debug(
            f'召回合并后,删除无用用户指标: {hitrate_5}, {mrr_5}, {hitrate_10}, {mrr_10}, {hitrate_20}, {mrr_20}, {hitrate_40}, {mrr_40}, {hitrate_50}, {mrr_50}'
        )

    df = df_useful_recall['user_id'].value_counts().reset_index()
    df.columns = ['user_id', 'cnt']
    log.debug(f"删除后平均每个用户召回数量：{df['cnt'].mean()}")

    log.debug(
        f"删除前标签分布: {recall_final[recall_final['label'].notnull()]['label'].value_counts()}"
        f"删除后标签分布: {df_useful_recall[df_useful_recall['label'].notnull()]['label'].value_counts()}"
    )

    # 保存到本地
    if mode == 'valid':
        df_useful_recall.to_pickle('user_data/data/offline/recall.pkl')
    else:
        df_useful_recall.to_pickle('user_data/data/online/recall.pkl')
