import numpy as np
import time
import rustlib

from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split


class AdaRank(BaseEstimator):
    """AdaRank algorithm"""

    def __init__(self, T=500, estop=10, verbose=False):
        self.T = T
        self.estop = estop
        self.verbose = verbose
        self.best_score = 0
        self.best_round = 0

    def fit(self, x, y):
        """Fit a model to the data"""

        qid = np.array(x[:,0])
        X = np.array(x[:,1:])

        if self.estop > 0:
            X, X_valid, y, y_valid, qid, qid_valid = train_test_split(X, y, qid, test_size=0.33, shuffle=True)

        # init values
        docs_per_qid = [X[qid==qi] for qi in np.unique(qid)]
        y_per_qid = [y[qid==qi] for qi in np.unique(qid)]
        self.pred_train = [np.zeros(len(X[qid==qi])) for qi in np.unique(qid)]
        idx_per_qid = [qi for qi in range(len(np.unique(qid)))]

        # initial values
        n_queries = np.unique(qid).shape[0]
        p = [np.ones(n_queries, dtype=np.float64) / n_queries]
        self.h = []
        self.alpha = []

        if self.verbose:
            print("Get WeakRankers")
        weak_rankers = rustlib.WeakRanker(docs_per_qid, y_per_qid)
        # trainings loop
        for t in range(self.T):
            if self.verbose:
                print(f"Round {t}, #queries {n_queries}")

            start = time.time()
            current_h, current_alpha = weak_rankers.get_h_alpha(p[t])
            self.h.append(current_h)
            self.alpha.append(current_alpha)
            end = time.time()
            if self.verbose:
                print("Get h and alpha time: ", end - start)

            # update p
            numerator = []
            denominator = 0
            cur_score_list = []
            start = time.time()
            for x_docs, y_docs, idx_docs in zip(docs_per_qid, y_per_qid, idx_per_qid):
                cur_score = rustlib.ndcg10(self.predict_proba_training(x_docs, t, idx_docs), y_docs)
                cur_score_list.append(cur_score)
                numerator.append(np.exp(-cur_score))
                denominator += np.exp(-cur_score)
            p.append(np.array(numerator)/denominator)
            end = time.time()
            if self.verbose:
                print("Get p time: ", end - start)

            if self.estop > 0:
                cur_score_list = []
                for qid in qid_valid:
                    pred = self.predict_proba(X_valid[qid_valid==qid], t)
                    cur_score_list.append(rustlib.ndcg10(pred, y_valid[qid_valid==qid]))
                cur_score = np.mean(cur_score_list)
                if self.best_score < cur_score:
                    self.best_score = cur_score
                    self.best_round = t
                if (t - self.best_round) > self.estop:
                    self.T = self.best_round
                    break

            if self.verbose:
                print(f"step:{t} alpha: {self.alpha[t]} score: {np.mean(cur_score_list)} best val_score: {self.best_score}")

    def predict_proba_training(self, X, t, qid):
        for xidx, xi in enumerate(X):
            self.pred_train[qid][xidx] += self.alpha[t] * xi[self.h[t]]
        return np.array(self.pred_train[qid])

    def predict_proba(self, X, t=None):
        """Make predictions"""
        if t == None: # here we are not in the training
            t = self.T
            X = X[:,1:]
        else:
            t = t + 1
        pred = np.zeros(X.shape[0])
        for ti in range(t):
            for xidx, xi in enumerate(X):
                pred[xidx] += self.alpha[ti] * xi[self.h[ti]]
        return np.array(pred)
