"""
This functions will be share when using all keras, pytorch and gluon,  so is
better to place then in a separate module
"""

import numpy as np
import math


def get_train_instances(train, n_items, n_neg, testNegatives, mode='pytorch'):
    user, item, labels = [],[],[]
    n_users = train.shape[0]
    for (u, i) in train.keys():
        # positive instance
        user.append(u)
        item.append(i)
        labels.append(1)
        # negative instances: we also need to make sure they are not in the
        # test dataset
        for t in range(n_neg):
            j = np.random.randint(n_items)
            while ((u, j) in train.keys()) or (j in testNegatives[u]):
                j = np.random.randint(n_items)
            user.append(u)
            item.append(j)
            labels.append(0)
    if mode == "keras":
        return np.array(user), np.array(item), np.array(labels)
    elif (mode == "pytorch") or (mode == "gluon"):
        return np.vstack([user,item,labels]).T


def get_test_instances(testRatings, testNegatives):
    testset = []
    for idx in range(len(testRatings)):
        rating = testRatings[idx]
        negItems = testNegatives[idx]
        u = rating[0]
        posItem = rating[1]
        items = np.array([posItem] + negItems)
        user = np.full(len(items), u, dtype = 'int32')
        labels = np.array([1] + 99*[0])
        userTestSet = np.vstack([user,items,labels]).T
        testset.append(userTestSet)
    return np.vstack(testset)


def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0


def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i+2)
    return 0