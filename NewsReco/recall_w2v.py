import argparse
import math
import os
import pickle
import random
import signal
import warnings
from collections import defaultdict
from random import shuffle

import multitasking
import numpy as np
import pandas as pd
from annoy import AnnoyIndex
from gensim.models import Word2Vec
from tqdm import tqdm

from utils import Logger, evaluate

warnings.filterwarnings('ignore')

max_threads = multitasking.config['CPU_CORES']
multitasking.set_max_threads(max_threads)
multitasking.set_engine('process')
signal.signal(signal.SIGINT, multitasking.killall)

seed = 2020
random.seed(seed)

# 命令行参数
parser = argparse.ArgumentParser(description='w2v 召回')
parser.add_argument('--mode', default='valid')
parser.add_argument('--logfile', default='test.log')

args = parser.parse_args()

mode = args.mode
logfile = args.logfile

# 初始化日志
os.makedirs('../user_data/log', exist_ok=True)
log = Logger(f'../user_data/log/{logfile}').logger
log.info(f'w2v 召回，mode: {mode}')

'''
除了使用人工规则从序列中提取相似度，我们还可以使用序列学习模型 Word2Vec 为新闻学习向量表示。
将用户的新闻点击序列作为句子喂到 Word2Vec 模型，然后选取和用户最近点击新闻最相似的关联新闻。
向量学习和寻找相似全部使用 gensim 库的 Word2Vec 实现。
'''
# df_click, 'user_id', 'click_article_id'
def word2vec(df_, f1, f2, model_path):
    df = df_.copy()
    tmp = df.groupby(f1, as_index=False)[f2].agg(
        {'{}_{}_list'.format(f1, f2): list})
    # sentenses 是将每个用户的点击历史点击新闻id按照从前往后的顺序存到一个list中  所以的用户形成一个总的list
    # 以下是5个用户的
    # [[300470, 16129, 160974], [160974], [183665, 181686], [160974, 202557], [160974, 160417]]
    sentences = tmp['{}_{}_list'.format(f1, f2)].values.tolist()
    del tmp['{}_{}_list'.format(f1, f2)]

    words = []
    #将每个用户的历史列表转为字符串
    for i in range(len(sentences)):
        #  只有转换成字符串才可以进行训练
        x = [str(x) for x in sentences[i]]
        sentences[i] = x
        words += x

    if os.path.exists(f'{model_path}/w2v.m'):
        model = Word2Vec.load(f'{model_path}/w2v.m')
    else:
        model = Word2Vec(sentences=sentences,
                         size=256, # size: 表示词向量的维度。
                         window=3, #window：决定了目标词会与多远距离的上下文产生关系。
                         min_count=1, # min_count: 设置最小的
                         sg=1, #sg: 如果是0，则是CBOW模型，是1则是Skip-Gram模型。
                         hs=0,
                         seed=seed,
                         negative=5,
                         workers=10, # workers: 表示训练时候的线程数量
                         iter=1) # iter: 训练时遍历整个数据集的次数
        model.save(f'{model_path}/w2v.m')

    article_vec_map = {}
    for word in set(words):
        if word in model:
            # key是新闻id，value是训练好的相应的新闻id的词向量 存起来
            article_vec_map[int(word)] = model[word]

    return article_vec_map


@multitasking.task
def recall(df_query, article_vec_map, article_index, user_item_dict,
           worker_id):
    data_list = []
    # 给验证集的用户召回
    for user_id, item_id in tqdm(df_query.values):
        rank = defaultdict(int)
        # 验证集的用户曾经点过的新闻
        interacted_items = user_item_dict[user_id]
        
        interacted_items = interacted_items[-1:]

        for item in interacted_items:
            article_vec = article_vec_map[item]
#If you set include_distances to True, it will return a 2 element tuple with two lists in it: 
#the second one containing all corresponding distances.
            # anno通过快速最近邻检索  通过余弦距离找到和当前向量最接近的100个相应的id
            # distances包含了这100个检索的 和 查询的向量 的余弦距离
            item_ids, distances = article_index.get_nns_by_vector(
                article_vec, 100, include_distances=True)
            # 通过该距离计算相似度
            sim_scores = [2 - distance for distance in distances]

            for relate_item, wij in zip(item_ids, sim_scores):
                if relate_item not in interacted_items:
                    rank.setdefault(relate_item, 0)
                    rank[relate_item] += wij

        sim_items = sorted(rank.items(), key=lambda d: d[1], reverse=True)[:50]
        item_ids = [item[0] for item in sim_items]
        item_sim_scores = [item[1] for item in sim_items]

        df_temp = pd.DataFrame()
        df_temp['article_id'] = item_ids
        df_temp['sim_score'] = item_sim_scores
        df_temp['user_id'] = user_id

        if item_id == -1:
            df_temp['label'] = np.nan
        else:
            df_temp['label'] = 0
            df_temp.loc[df_temp['article_id'] == item_id, 'label'] = 1

        df_temp = df_temp[['user_id', 'article_id', 'sim_score', 'label']]
        df_temp['user_id'] = df_temp['user_id'].astype('int')
        df_temp['article_id'] = df_temp['article_id'].astype('int')

        data_list.append(df_temp)

    df_data = pd.concat(data_list, sort=False)

    os.makedirs('../user_data/tmp/w2v', exist_ok=True)
    df_data.to_pickle('../user_data/tmp/w2v/{}.pkl'.format(worker_id))


if __name__ == '__main__':
    if mode == 'valid':
        df_click = pd.read_pickle('../user_data/data/offline/click.pkl')
        df_query = pd.read_pickle('../user_data/data/offline/query.pkl')

        os.makedirs('../user_data/data/offline', exist_ok=True)
        os.makedirs('../user_data/model/offline', exist_ok=True)

        w2v_file = '../user_data/data/offline/article_w2v.pkl'
        model_path = '../user_data/model/offline'
    else:
        df_click = pd.read_pickle('../user_data/data/online/click.pkl')
        df_query = pd.read_pickle('../user_data/data/online/query.pkl')

        os.makedirs('../user_data/data/online', exist_ok=True)
        os.makedirs('../user_data/model/online', exist_ok=True)

        w2v_file = '../user_data/data/online/article_w2v.pkl'
        model_path = '../user_data/model/online'

    log.debug(f'df_click shape: {df_click.shape}')
    log.debug(f'{df_click.head()}')
    # 得到每个文章id对应的词向量
    article_vec_map = word2vec(df_click, 'user_id', 'click_article_id',
                               model_path)
    f = open(w2v_file, 'wb')
    # 将得到的词向量 保存至文件中
    pickle.dump(article_vec_map, f)
    f.close()

    # 说白了就是先将加载进来的vector进行相似临近计算，然后生成一个树形结构的索引，这样查找速度会变得很快，只不过会牺牲一定的近似精度。
    # 将 embedding 建立索引
    article_index = AnnoyIndex(256, 'angular') #metric='angular'表示使用 angular（余弦）距离度量来计算簇和哈希。
    article_index.set_seed(2020)
    ##加载article_id和向量映射，添加到annoyIndex
    for article_id, emb in tqdm(article_vec_map.items()):
        article_index.add_item(article_id, emb)
    # tree_num设置为100，在内存允许的情况下，越大越好
    article_index.build(100)

    user_item_ = df_click.groupby('user_id')['click_article_id'].agg(
        lambda x: list(x)).reset_index()
    user_item_dict = dict(
        zip(user_item_['user_id'], user_item_['click_article_id']))

    # 召回
    n_split = max_threads
    all_users = df_query['user_id'].unique()
    shuffle(all_users)
    total = len(all_users)
    n_len = total // n_split

    # 清空临时文件夹
    for path, _, file_list in os.walk('../tmp/w2v'):
        for file_name in file_list:
            os.remove(os.path.join(path, file_name))

    for i in range(0, total, n_len):
        part_users = all_users[i:i + n_len]
        df_temp = df_query[df_query['user_id'].isin(part_users)]
        recall(df_temp, article_vec_map, article_index, user_item_dict, i)

    multitasking.wait_for_tasks()
    log.info('合并任务')

    df_data = pd.DataFrame()
    # 使用多线程召回 结果被存到了多个文件中
    for path, _, file_list in os.walk('../user_data/tmp/w2v'):
        for file_name in file_list:
            df_temp = pd.read_pickle(os.path.join(path, file_name))
            df_data = df_data.append(df_temp)

    # 合并完后 必须加，对其进行排序
    df_data = df_data.sort_values(['user_id', 'sim_score'],
                                  ascending=[True,
                                             False]).reset_index(drop=True)
    log.debug(f'df_data.head: {df_data.head()}')

    # 计算召回指标
    if mode == 'valid':
        log.info(f'计算召回指标')
        # user_id 唯一值的个数 即 多少个验证集的用户
        total = df_query[df_query['click_article_id'] != -1].user_id.nunique()

        hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50 = evaluate(
            df_data[df_data['label'].notnull()], total)

        log.debug(
            f'w2v: {hitrate_5}, {mrr_5}, {hitrate_10}, {mrr_10}, {hitrate_20}, {mrr_20}, {hitrate_40}, {mrr_40}, {hitrate_50}, {mrr_50}'
        )
    # 保存召回结果
    if mode == 'valid':
        df_data.to_pickle('../user_data/data/offline/recall_w2v.pkl')
    else:
        df_data.to_pickle('../user_data/data/online/recall_w2v.pkl')
