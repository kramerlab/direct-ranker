# code adapted from https://github.com/lezzago/LambdaMart/blob/master/lambdamart.py

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import BaseEstimator
from multiprocessing import Pool


def dcg(scores):
	"""
		Returns the DCG value of the list of scores.
		Parameters
		----------
		scores : list
			Contains labels in a certain ranked order
		
		Returns
		-------
		DCG_val: int
			This is the value of the DCG on the given scores
	"""
	return np.sum([
						(np.power(2, scores[i]) - 1) / np.log2(i + 2)
						for i in range(len(scores))
					])

def dcg_k(scores, k):
	"""
		Returns the DCG value of the list of scores and truncates to k values.
		Parameters
		----------
		scores : list
			Contains labels in a certain ranked order
		k : int
			In the amount of values you want to only look at for computing DCG
		
		Returns
		-------
		DCG_val: int
			This is the value of the DCG on the given scores
	"""
	return np.sum([
						(np.power(2, scores[i]) - 1) / np.log2(i + 2)
						for i in range(len(scores[:k]))
					])

def ideal_dcg(scores):
	"""
		Returns the Ideal DCG value of the list of scores.
		Parameters
		----------
		scores : list
			Contains labels in a certain ranked order
		
		Returns
		-------
		Ideal_DCG_val: int
			This is the value of the Ideal DCG on the given scores
	"""
	scores = [score for score in sorted(scores)[::-1]]
	return dcg(scores)

def ideal_dcg_k(scores, k):
	"""
		Returns the Ideal DCG value of the list of scores and truncates to k values.
		Parameters
		----------
		scores : list
			Contains labels in a certain ranked order
		k : int
			In the amount of values you want to only look at for computing DCG
		
		Returns
		-------
		Ideal_DCG_val: int
			This is the value of the Ideal DCG on the given scores
	"""
	scores = [score for score in sorted(scores)[::-1]]
	return dcg_k(scores, k)

def single_dcg(scores, i, j):
	"""
		Returns the DCG value at a single point.
		Parameters
		----------
		scores : list
			Contains labels in a certain ranked order
		i : int
			This points to the ith value in scores
		j : int
			This sets the ith value in scores to be the jth rank
		
		Returns
		-------
		Single_DCG: int
			This is the value of the DCG at a single point
	"""
	return (np.power(2, scores[i]) - 1) / np.log2(j + 2)

def compute_lambda(args):
	"""
		Returns the lambda and w values for a given query.
		Parameters
		----------
		args : zipped value of true_scores, predicted_scores, good_ij_pairs, idcg, query_key
			Contains a list of the true labels of documents, list of the predicted labels of documents,
			i and j pairs where true_score[i] > true_score[j], idcg values, and query keys.
		
		Returns
		-------
		lambdas : numpy array
			This contains the calculated lambda values
		w : numpy array
			This contains the computed w values
		query_key : int
			This is the query id these values refer to
	"""

	true_scores, predicted_scores, good_ij_pairs, idcg, query_key = args
	num_docs = len(true_scores)
	sorted_indexes = np.argsort(predicted_scores)[::-1]
	rev_indexes = np.argsort(sorted_indexes)
	true_scores = true_scores[sorted_indexes]
	predicted_scores = predicted_scores[sorted_indexes]

	lambdas = np.zeros(num_docs)
	w = np.zeros(num_docs)

	single_dcgs = {}
	for i,j in good_ij_pairs:
		if (i,i) not in single_dcgs:
			single_dcgs[(i,i)] = single_dcg(true_scores, i, i)
		single_dcgs[(i,j)] = single_dcg(true_scores, i, j)
		if (j,j) not in single_dcgs:
			single_dcgs[(j,j)] = single_dcg(true_scores, j, j)
		single_dcgs[(j,i)] = single_dcg(true_scores, j, i)


	for i,j in good_ij_pairs:
		z_ndcg = abs(single_dcgs[(i,j)] - single_dcgs[(i,i)] + single_dcgs[(j,i)] - single_dcgs[(j,j)]) / idcg
		rho = 1 / (1 + np.exp(predicted_scores[i] - predicted_scores[j]))
		rho_complement = 1.0 - rho
		lambda_val = z_ndcg * rho
		lambdas[i] += lambda_val
		lambdas[j] -= lambda_val

		w_val = rho * rho_complement * z_ndcg
		w[i] += w_val
		w[j] += w_val

	return lambdas[rev_indexes], w[rev_indexes], query_key

def group_queries(q):
	"""
		Returns a dictionary that groups the documents by their query ids.
		Parameters
		----------
		q : Numpy array of lists
			Contains a list query index
		
		Returns
		-------
		query_indexes : dictionary
			The keys were the different query ids and teh values were the indexes in the training data that are associated of those keys.
	"""
	query_indexes = {}
	index = 0
	for record in q:
		query_indexes.setdefault(record, [])
		query_indexes[record].append(index)
		index += 1
	return query_indexes

def get_pairs(scores):
	"""
		Returns pairs of indexes where the first value in the pair has a higher score than the second value in the pair.
		Parameters
		----------
		scores : list of int
			Contain a list of numbers
		
		Returns
		-------
		query_pair : list of pairs
			This contains a list of pairs of indexes in scores.
	"""

	query_pair = []
	for query_scores in scores:
		temp = sorted(query_scores, reverse=True)
		pairs = []
		for i in range(len(temp)):
			for j in range(len(temp)):
				if temp[i] > temp[j]:
					pairs.append((i,j))
		query_pair.append(pairs)
	return query_pair

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
		predicted_scores = np.zeros(len(x))
		query_indexes = group_queries(q)
		query_keys = query_indexes.keys()
		true_scores = [y[query_indexes[query]] for query in query_keys]
		good_ij_pairs = get_pairs(true_scores)

		# ideal dcg calculation
		idcg = [ideal_dcg(scores) for scores in true_scores]

		for k in range(self.number_of_trees):
			print('Tree %d' % (k))
			lambdas = np.zeros(len(predicted_scores))
			w = np.zeros(len(predicted_scores))
			pred_scores = [predicted_scores[query_indexes[query]] for query in query_keys]

			pool = Pool()
			for lambda_val, w_val, query_key in pool.map(compute_lambda, zip(true_scores, pred_scores, good_ij_pairs, idcg, query_keys), chunksize=1):
				indexes = query_indexes[query_key]
				lambdas[indexes] = lambda_val
				w[indexes] = w_val
			pool.close()

			# Sklearn implementation of the tree			
			tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
			tree.fit(x, lambdas)
			self.trees.append(tree)
			prediction = tree.predict(x)
			predicted_scores += prediction * self.learning_rate

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
		data = data[:,1:]
		results = 0
		for tree in self.trees:
			results += self.learning_rate * tree.predict(data)
		return results
