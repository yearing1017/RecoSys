import numpy as np
import argparse
import math


def get_data(path,num_users,num_items,num_total_ratings,train_ratio):

    fp = open(path + "ratings.dat") #打开文件

    user_train_set = set() #用户训练集
    user_test_set = set() #用户测试集
    item_train_set = set() #物品训练集
    item_test_set = set() #物品测试集

    train_r = np.zeros((num_users, num_items)) #训练评分矩阵
    test_r = np.zeros((num_users, num_items)) #测试评分矩阵

    train_mask_r = np.zeros((num_users, num_items)) #用来记录在训练集中已经评过分的电影
    test_mask_r = np.zeros((num_users, num_items)) #用来记录在测试集中已经评过分的电影

    random_perm_idx = np.random.permutation(num_total_ratings) #将评分编号进行打乱
    train_idx = random_perm_idx[0:int(num_total_ratings * train_ratio)]  #将编号靠前的记为训练集编号
    test_idx = random_perm_idx[int(num_total_ratings * train_ratio):] #将编号靠后的记为测试集编号

    lines = fp.readlines() #按照行来读取文件

    ''' Train '''
    for itr in train_idx:
        line = lines[itr]
        user, item, rating, _ = line.split("::") #dat文件是用::隔开的
        user_idx = int(user) - 1 #dat文件中的ID是以1开始的
        item_idx = int(item) - 1
        train_r[user_idx, item_idx] = int(rating) #构造训练集评分表，用户id为user_idx的用户对电影编号为item_idx的电影打了rating的分数
        train_mask_r[user_idx, item_idx] = 1 #代表用户id为user_idx的用户对电影编号为item_idx的电影进行了评分

        user_train_set.add(user_idx) #构造用户训练集
        item_train_set.add(item_idx) #构造物品训练集

    ''' Test '''
    for itr in test_idx:
        line = lines[itr]
        user, item, rating, _ = line.split("::")
        user_idx = int(user) - 1
        item_idx = int(item) - 1
        test_r[user_idx, item_idx] = int(rating)
        test_mask_r[user_idx, item_idx] = 1

        user_test_set.add(user_idx) #构造用户测试集
        item_test_set.add(item_idx) #构造物品测试集


    return train_r,train_mask_r,test_r,test_mask_r,user_train_set,item_train_set,user_test_set,item_test_set