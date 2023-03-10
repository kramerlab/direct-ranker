# code adapted from https://github.com/discobot/LambdaMart/blob/master/mart.py

import math
import numpy as np
import math
from optparse import OptionParser
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import BaseEstimator
from collections import defaultdict
from copy import deepcopy
from multiprocessing import Pool
from itertools import chain
import time

class Ensemble:
    def __init__(self, rate):
        self.trees = []
        self.rate = rate

    def __len__(self):
        return len(self.trees)

    def add(self, tree):
        self.trees.append(tree)

    def eval_one(self, object):
        return self.eval([object])[0]

    def eval(self, objects):
        results = np.zeros(len(objects))
        for tree in self.trees:
            results += tree.predict(objects) * self.rate
        return results

    def remove(self, number):
        self.trees = self.trees[:-number]


def groupby(score, query):
    result = []
    this_query = None
    for s, q in zip(score, query):
        if q != this_query:
            result.append([])
            this_query = q
        result[-1].append(s)
    result = map(np.array, result)
    return result


def point_dcg(arg):
    i, label = arg
    return (2 ** label - 1) / math.log(i + 2, 2)


def dcg(scores):
    return sum(map(point_dcg, enumerate(scores)))


def ndcg(page, k=10):
    model_top = page[:k]

    true_top = np.array([])
    if len(page) > 10:
        true_top = np.partition(page, -10)[-k:]
        true_top.sort()
    else:
        true_top = np.sort(page)
    true_top = true_top[::-1]


    max_dcg = dcg(true_top)
    model_dcg = dcg(model_top)

    if max_dcg == 0:
        return 1

    return model_dcg / max_dcg


def score(prediction, true_score, query, k=10):
    true_pages = groupby(true_score, query)
    model_pages = groupby(prediction, query)

    total_ndcg = []

    for true_page, model_page in zip(true_pages, model_pages):
        page = true_page[np.argsort(model_page)[::-1]]
        total_ndcg.append(ndcg(page, k))

    return sum(total_ndcg) / len(total_ndcg)


def query_lambdas(page, k=10):
    true_page, model_page = page
    worst_order = np.argsort(true_page)

    true_page = true_page[worst_order]
    model_page = model_page[worst_order]
 

    model_order = np.argsort(model_page)

    idcg = dcg(np.sort(true_page)[-10:][::-1])

    size = len(true_page)
    position_score = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            position_score[model_order[i], model_order[j]] = \
                point_dcg((model_order[j], true_page[model_order[i]]))

    lambdas = np.zeros(size)

    for i in range(size):
        for j in range(size):
                if true_page[i] > true_page[j]:

                    delta_dcg  = position_score[i][j] - position_score[i][i]
                    delta_dcg += position_score[j][i] - position_score[j][j]

                    delta_ndcg = abs(delta_dcg / idcg)

                    rho = 1 / (1 + math.exp(model_page[i] - model_page[j]))

                    lam = rho * delta_ndcg

                    lambdas[j] -= lam
                    lambdas[i] += lam
    return lambdas


def compute_lambdas(prediction, true_score, query, k=10):
    true_pages = groupby(true_score, query)
    model_pages = groupby(prediction, query)

    pool = Pool()
    lambdas = pool.map(query_lambdas, zip(true_pages, model_pages))
    return list(chain(*lambdas))


def mart_responces(prediction, true_score):
    return true_score - prediction


class LambdaMart(BaseEstimator):

	def __init__(self, number_of_trees=5, learning_rate=0.1, max_depth=50, min_samples_split=2):
		"""
		This is the constructor for the LambdaMART object.
		Parameters
		----------
		number_of_trees : int (default: 5)
			Number of trees LambdaMART goes through
		learning_rate : float (default: 0.1)
			Rate at which we update our prediction with each tree
		"""

		self.number_of_trees = number_of_trees
		self.learning_rate = learning_rate
		self.max_depth = max_depth
		self.min_samples_split = min_samples_split
		self.trees = []

	def fit(self, x, y):
		"""
		Fits the model on the training data.
		"""

		q = x[:,0]
		x = x[:,1:]

		scores = y
		queries = q
		features = x

		ensemble = Ensemble(self.learning_rate)

		print("Training starts...")
		model_output = np.zeros(len(features))

		for i in range(self.number_of_trees):
			print(" Iteration: " + str(i + 1))

			# Compute psedo responces (lambdas)
			# witch act as training label for document
			start = time.time()
			print("  --generating labels")
			lambdas = compute_lambdas(model_output, scores, queries, k=10)
			print("  --done", str(time.time() - start) + " sec")
	
			# create tree and append it to the model
			print("  --fitting tree")
			start = time.time()
			tree = DecisionTreeRegressor(max_depth=self.max_depth)
			tree.fit(features, lambdas)

			print("  --done", str(time.time() - start) + " sec")
			print("  --adding tree to ensemble")
			ensemble.add(tree)

			# update model score
			print("  --generating step prediction")
			prediction = tree.predict(features)

			print("  --updating full model output")
			model_output += self.learning_rate * prediction

			# train_score
			start = time.time()
			print("  --scoring on train")
			train_score = score(model_output, scores, queries, 10)
			print("  --iteration train score " + str(train_score) + ", took " + str(time.time() - start) + "sec to calculate")

		# finishing up
		print("Finished sucessfully.")
		print("------------------------------------------------")
		self.ensemble = ensemble

	def predict_proba(self, data):
		"""
		Predicts the scores for the test dataset.
		Parameters
		----------
		data : Numpy array of documents of a single query
		
		Returns
		-------
		predicted_scores : Numpy array of scores
			This contains an array or the predicted scores for the documents.
		"""
		return self.ensemble.eval(data[:,1:])
