import pandas as pd
import numpy as np
import argparse
import os
'''数据简介
train.rating: Train file.
Each Line is a training instance: userID\t itemID\t rating\t timestamp (if have)

test.rating: Test file (positive instances).
Each Line is a testing instance: userID\t itemID\t rating\t timestamp (if have)

test.negative: Test file (negative instances).
Each line corresponds to the line of test.rating, containing 99 negative samples.
Each line is in the format: (userID,itemID)\t negativeItemID1\t negativeItemID2 ...
此处的(userID,itemID)为测试集中的组合；后面为每个用户从未vote的99部电影
'''


def prepare_ml1m(input_dir, input_fname, out_dir, asdate=False):

	# 四列内容在dat文件中以::作为间隔
	ratings = pd.read_csv(os.path.join(input_dir, input_fname),
		sep='::', engine='python',
		names=['userid', 'itemid', 'rating', 'timestamp'])

	# python starts counting from 0
	ratings['userid'] = ratings['userid'] - 1
	ratings['itemid'] = ratings['itemid'] - 1

	if asdate:
		ratings.timestamp = pd.to_datetime(ratings.timestamp, unit='s')

	# check that userid and itemid contain consecutive numbers. Important for the embeddings
	users_ok = (sorted(ratings.userid.unique()) == \
		list(range(min(ratings.userid.unique()), max(ratings.userid.unique())+1)))
	items_ok = (sorted(ratings.itemid.unique()) == \
		list(range(min(ratings.itemid.unique()), max(ratings.itemid.unique())+1)))
	# print(users_ok)
	# print(items_ok)
	# --
	# True
	# False

	#itemsids are not continuous, which need to be for the embeddings lookup
	item_map = dict(
		zip(sorted(ratings.itemid.unique()), range(len(ratings.itemid.unique())))
		)
	ratings['itemid'] = ratings.itemid.replace(item_map)

	# sort by userid and timestamp
	ratings.sort_values(['userid','timestamp'], ascending=[True,True], inplace=True)
	ratings.reset_index(inplace=True, drop=True)

	# use last ratings for testing and all the previous for training
	test_ratings = ratings.groupby('userid').tail(1)
	train_ratings = pd.merge(ratings, test_ratings, on=['userid','itemid'],
		how='outer', suffixes=('', '_y'))
	train_ratings = train_ratings[train_ratings.rating_y.isnull()]
	test_ratings = test_ratings[['userid','itemid','rating', 'timestamp']]
	train_ratings = train_ratings[['userid','itemid','rating', 'timestamp']]

	# select 99 random movies per user that were never rated
	all_items = ratings.itemid.unique()
	negative = (ratings.groupby("userid")['itemid']
	    .apply(list)
	    .reset_index()
	    )
	np.random.seed=1981
	negative['negative'] = negative.itemid.apply(
		lambda x: np.random.choice(np.setdiff1d(all_items, x), 99))

	negative.drop('itemid', axis=1, inplace=True)
	negative= test_ratings.merge(negative, on='userid')
	negative['positive'] = negative[['userid', 'itemid']].apply(tuple, axis=1)
	negative.drop(['userid','itemid', 'rating', 'timestamp'], axis=1, inplace=True)
	negative = negative[['positive','negative']]
	negative[['item_n'+str(i) for i in range(99)]] = \
		pd.DataFrame(negative.negative.values.tolist(), index= negative.index)
	negative.drop('negative', axis=1, inplace=True)

	# Save the set up
	train_ratings.to_csv(os.path.join(out_dir,'ml-1m.train.rating'), sep="\t", index=False)
	test_ratings.to_csv(os.path.join(out_dir,'ml-1m.test.rating'), sep="\t", index=False)
	negative.to_csv(os.path.join(out_dir,'ml-1m.test.negative'), sep="\t", index=False)

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description="prepare ml-1m dataset for neural CF")

	parser.add_argument("--input_dir",type=str, default="ml-1m-data",)
	parser.add_argument("--input_fname",type=str, default="ratings.dat",)
	parser.add_argument("--out_dir",type=str, default="data-after-prepare",)
	parser.add_argument("--asdate", action='store_true')
	args = parser.parse_args()

	prepare_ml1m(
		args.input_dir,
		args.input_fname,
		args.out_dir)