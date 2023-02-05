import scipy.sparse as sp
import pandas as pd
import numpy as np

class Dataset(object):
    def __init__(self, path):
        """
        Constructor to build the datasets
        """
        self.trainMatrix = self.load_rating_file_as_matrix(path + ".train.rating")
        self.testRatings = self.load_rating_file_as_list(path + ".test.rating")
        self.testNegatives = self.load_negative_file(path + ".test.negative")
        assert len(self.testRatings) == len(self.testNegatives)

        self.num_users, self.num_items = self.trainMatrix.shape

    def load_rating_file_as_list(self, filename):
        df = pd.read_csv(filename, sep="\t")
        ratingList = list(zip(df.userid.tolist(), df.itemid.tolist()))
        return ratingList

    def load_negative_file(self, filename):
        df = pd.read_csv(filename, sep="\t")
        negativeList = df.iloc[:, 1:].values.tolist()
        return negativeList

    def load_rating_file_as_matrix(self, filename):
        '''
        Read .rating file and Return dok matrix.
        The first line of .rating file is: num_users\t num_items
        '''
        df = pd.read_csv(filename, sep="\t")
        num_users = df.userid.max()
        num_items = df.itemid.max()
        mat = sp.dok_matrix((num_users+1, num_items+1), dtype=np.float32)
        interactions = df[['userid','itemid']].values.tolist()
        for user, item in interactions:
                mat[user, item] = 1.
        return mat