# import packages
import time, math, os
from tqdm import tqdm
import gc
import pickle
import random
from datetime import datetime
from operator import itemgetter
import numpy as np
import pandas as pd
import warnings
from collections import defaultdict
import collections
warnings.filterwarnings('ignore')

# data_path = './data_raw/'
data_path = '/home/admin/jupyter/data/' # 所有数据文件
save_path = '/home/admin/jupyter/temp_results/'  # 暂存数据路径

# debug模式：从训练集中划出一部分数据来调试代码 这里的目的是为了跑通数据
def get_all_click_sample(data_path, sample_nums=10000):
    """
        训练集中采样一部分数据调试
        data_path: 原数据的存储路径
        sample_nums: 采样数目（这里由于机器的内存限制，可以采样用户做）
    """
    all_click = pd.read_csv(data_path + 'train_click_log.csv')
    all_user_ids = all_click.user_id.unique()

    sample_user_ids = np.random.choice(all_user_ids, size=sample_nums, replace=False) 
    all_click = all_click[all_click['user_id'].isin(sample_user_ids)]
    
    all_click = all_click.drop_duplicates((['user_id', 'click_article_id', 'click_timestamp']))
    return all_click

# 读取点击数据，这里分成线上和线下，如果是为了获取线上提交结果应该讲测试集中的点击数据合并到总的数据中
# 如果是为了线下验证模型的有效性或者特征的有效性，可以只使用训练集，训练时应指定为线下模式
def get_all_click_df(data_path='./data_raw/', offline=True):
    if offline:
        all_click = pd.read_csv(data_path + 'train_click_log.csv')
    else:
        trn_click = pd.read_csv(data_path + 'train_click_log.csv')
        tst_click = pd.read_csv(data_path + 'testA_click_log.csv')

        all_click = trn_click.append(tst_click)
    # 去掉重复的数据
    all_click = all_click.drop_duplicates((['user_id', 'click_article_id', 'click_timestamp']))
    return all_click

# 根据点击时间获取用每个户的点击文章序列   {user1: [(item1, time1), (item2, time2)..]...}
def get_user_item_time(click_df):
    
    click_df = click_df.sort_values('click_timestamp')
    
    def make_item_time_pair(df):
        return list(zip(df['click_article_id'], df['click_timestamp']))
    
    user_item_time_df = click_df.groupby('user_id')['click_article_id', 'click_timestamp'].apply(lambda x: make_item_time_pair(x))\
                                                            .reset_index().rename(columns={0: 'item_time_list'})
    user_item_time_dict = dict(zip(user_item_time_df['user_id'], user_item_time_df['item_time_list']))
    
    return user_item_time_dict

# 获取近期点击最多的k篇文章
def get_item_topk_click(click_df, k):
    topk_click = click_df['click_article_id'].value_counts().index[:k]
    return topk_click


"""
    文章与文章之间的相似性矩阵计算
    :param df: 数据表
    :item_created_time_dict:  文章创建时间的字典
    return : 文章与文章的相似性矩阵
"""

def itemcf_sim(df):   
    user_item_time_dict = get_user_item_time(df)
    
    # 计算物品相似度
    i2i_sim = {}
    item_cnt = defaultdict(int)
    for user, item_time_list in tqdm(user_item_time_dict.items()):
        # 在基于商品的协同过滤优化的时候可以考虑时间因素
        for i, i_click_time in item_time_list:
            item_cnt[i] += 1
            i2i_sim.setdefault(i, {})
            for j, j_click_time in item_time_list:
                if(i == j):
                    continue
                i2i_sim[i].setdefault(j, 0)
                
                i2i_sim[i][j] += 1 / math.log(len(item_time_list) + 1)
                
    i2i_sim_ = i2i_sim.copy()
    for i, related_items in i2i_sim.items():
        for j, wij in related_items.items():
            i2i_sim_[i][j] = wij / math.sqrt(item_cnt[i] * item_cnt[j])
    
    # 将得到的相似性矩阵保存到本地
    #pickle.dump(i2i_sim_, open(save_path + 'itemcf_i2i_sim.pkl', 'wb'))
    
    return i2i_sim_

# 基于商品的召回i2i
"""
    基于文章协同过滤的召回
    :param user_id: 用户id
    :param user_item_time_dict: 字典, 根据点击时间获取用户的点击文章序列   {user1: [(item1, time1), (item2, time2)..]...}
    :param i2i_sim: 字典，文章相似性矩阵
    :param sim_item_topk: 整数， 选择与当前文章最相似的前k篇文章
    :param recall_item_num: 整数， 最后的召回文章数量
    :param item_topk_click: 列表，点击次数最多的文章列表，用户召回补全        
    return: 召回的文章列表 {item1:score1, item2: score2...}
"""
def item_based_recommend(user_id, user_item_time_dict, i2i_sim, sim_item_topk, recall_item_num, item_topk_click):
    
    # 获取用户历史交互的文章
    user_hist_items = user_item_time_dict[user_id]
    user_hist_items_ = {user_id for user_id, _ in user_hist_items}
    
    item_rank = {}
    for loc, (i, click_time) in enumerate(user_hist_items):
        for j, wij in sorted(i2i_sim[i].items(), key=lambda x: x[1], reverse=True)[:sim_item_topk]:
            if j in user_hist_items_:
                continue
                
            item_rank.setdefault(j, 0)
            item_rank[j] +=  wij
    
    # 不足10个，用热门商品补全
    if len(item_rank) < recall_item_num:
        for i, item in enumerate(item_topk_click):
            if item in item_rank.items(): # 填充的item应该不在原来的列表中
                continue
            item_rank[item] = - i - 100 # 随便给个负数就行
            if len(item_rank) == recall_item_num:
                break
    
    item_rank = sorted(item_rank.items(), key=lambda x: x[1], reverse=True)[:recall_item_num]
        
    return item_rank


if __name__ == '__main__':
    # 全量训练集
    all_click_df = get_all_click_df(data_path, offline=False)
    # 物品相似矩阵
    i2i_sim = itemcf_sim(all_click_df)
    
    # 用字典存储每个用户的召回结果
    user_recall_items_dict = collections.defaultdict(dict)

    # 获取 用户 - 文章 - 点击时间的字典
    user_item_time_dict = get_user_item_time(all_click_df)

    # 取文章相似度
    i2i_sim = pickle.load(open(save_path + 'itemcf_i2i_sim.pkl', 'rb'))

    # 相似文章的数量
    sim_item_topk = 10

    # 召回文章数量
    recall_item_num = 10

    # 召回热度TopK数量的文章
    item_topk_click = get_item_topk_click(all_click_df, k=50)
    # 对每个用户做召回
    for user in tqdm(all_click_df['user_id'].unique()):
        user_recall_items_dict[user] = item_based_recommend(user, user_item_time_dict, i2i_sim, 
                                                            sim_item_topk, recall_item_num, item_topk_click)

    # 将字典的形式转换成df
    user_item_score_list = []

    for user, items in tqdm(user_recall_items_dict.items()):
        for item, score in items:
            user_item_score_list.append([user, item, score])

    recall_df = pd.DataFrame(user_item_score_list, columns=['user_id', 'click_article_id', 'pred_score'])

    # 获取测试集
    tst_click = pd.read_csv(data_path + 'testA_click_log.csv')
    tst_users = tst_click['user_id'].unique()

    # 从所有的召回数据中将测试集中的用户选出来
    tst_recall = recall_df[recall_df['user_id'].isin(tst_users)]