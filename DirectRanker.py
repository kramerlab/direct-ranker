import numpy as np
import tensorflow as tf
import pickle
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split

from supplementary_code_direct_ranker.helpers import nDCG_cls

import time


def default_weight_function(w1, w2):
    return w1 * w2


class directRanker(BaseEstimator):
    """
    Constructor
    :param hidden_layers: List containing the numbers of neurons in the layers for feature
    :param feature_activation: tf function for the feature part of the net
    :param ranking_activation: tf function for the ranking part of the net
    :param feature_bias: boolean value if the feature part should contain a bias
    :param kernel_initializer: tf kernel_initializer
    :param dtype: dtype used in each layer
    :param cost: cost function for the directRanker
    :param weight_function: weight_function for the documents weights
    :param start_batch_size: start value for increasing the sample size
    :param end_batch_size: end value for increasing the sample size
    :param learning_rate: learning rate for the optimizer
    :param max_steps: total training steps
    :param learning_rate_step_size: factor for increasing the learning rate
    :param learning_rate_decay_factor: factor for increasing the learning rate
    :param optimizer: tf optimizer object
    :param print_step: for which step the script should print out the cost for the current batch
    :param feature_func: additional feature_function for the feature part of the net
    :param weights: boolean if weights are passed in the fit
    :param end_qids: end value for increasing the query size
    :param start_qids: start value for increasing the query size
    """

    def __init__(self,
                 hidden_layers=[10],
                 feature_activation=tf.nn.tanh,
                 ranking_activation=tf.nn.tanh,
                 feature_bias=True,
                 kernel_initializer=tf.random_normal_initializer(),
                 dtype=tf.float32,
                 cost=None,
                 weight_function=None,
                 start_batch_size=100,
                 end_batch_size=10000,
                 learning_rate=0.01,
                 max_steps=10000,
                 learning_rate_step_size=500,
                 learning_rate_decay_factor=0.944,
                 optimizer=tf.train.AdamOptimizer,
                 print_step=0,
                 feature_func=None,
                 feature_func_nn0_1=None,
                 weights=False,
                 end_qids=300,
                 start_qids=10,
                 weight_regularization=0.,
                 dropout=0.,
                 input_dropout=0.,
                 early_stopping=False,
                 validation_size=0.2,
                 stop_scorer=nDCG_cls,
                 lookback=10,
                 stop_delta=0.001,
                 random_seed=None,
                 stop_start=None,
                 name="DirectRanker",
                 ):

        self.hidden_layers = hidden_layers
        self.feature_activation = feature_activation
        self.ranking_activation = ranking_activation
        self.feature_bias = feature_bias
        self.kernel_initializer = kernel_initializer
        self.dtype = dtype
        self.cost = cost
        self.weight_function = weight_function
        self.start_batch_size = start_batch_size
        self.end_batch_size = end_batch_size
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.learning_rate_step_size = learning_rate_step_size
        self.learning_rate_decay_factor = learning_rate_decay_factor
        self.optimizer = optimizer
        self.print_step = print_step
        self.feature_func = feature_func
        self.weights = weights
        self.end_qids = end_qids
        self.start_qids = start_qids
        self.weight_func = weight_function
        self.feature_func_nn0_1 = feature_func_nn0_1

        self.weight_regularization = weight_regularization
        self.dropout = dropout
        self.input_dropout = input_dropout
        self.early_stopping = early_stopping
        self.validation_size = validation_size
        self.random_seed = random_seed
        self.stop_scorer = stop_scorer
        self.lookback = lookback
        self.stop_delta = stop_delta
        self.name = name

        if stop_start is None:
            self.stop_start = int(self.max_steps / 2)
        else:
            self.stop_start = stop_start

        self.should_drop = None

        self.x0 = None
        self.x1 = None
        self.y0 = None
        self.w0 = None
        self.num_features = None
        self.sess = None
        self.num_hidden_layers = len(hidden_layers)

    def _to_dict(self, features, real_classes, weights):
        """
        A little function for preprocessing the data
        :param features: documents of a query
        :param real_classes: class leables of the documents
        :param weights: weights of the documents
        :return: dict with class as key and a list of queries
        """
        d = {}
        for i in range(len(features)):
            if real_classes[i, 0] in d.keys():
                if weights is None:
                    d[real_classes[i, 0]].append(features[i])
                else:
                    d[real_classes[i, 0]].append(np.concatenate((features[i], [weights[i]])))
            else:
                if weights is None:
                    d.update({real_classes[i, 0]: [features[i]]})
                else:
                    d.update({real_classes[i, 0]: [np.concatenate((features[i], [weights[i]]))]})
        for k in d.keys():
            d[k] = np.array(d[k])
        return d

    def _comparator(self, x1, x2):
        """
        :param x1: list of documents
        :param x2: list of documents
        :return: cmp value for sorting the query
        """
        res = self.evaluate(x1[:-1], x2[:-1])
        if res < 0:
            return -1
        elif res > 0:
            return 1
        return 0

    def _build_model(self):
        """
        This function builds the directRanker with the values specified in the constructor
        :return:
        """
        if self.weight_function is None:
            self.weight_function = default_weight_function

        tf.reset_default_graph()

        # Placeholders for the inputs
        self.x0 = tf.placeholder(
            shape=[None, self.num_features],
            dtype=self.dtype,
            name="x0"
        )
        self.x1 = tf.placeholder(
            shape=[None, self.num_features],
            dtype=self.dtype,
            name="x1"
        )
        # Placeholder for the real classes
        self.y0 = tf.placeholder(
            shape=[None, 1],
            dtype=self.dtype,
            name="y0"
        )
        # Placeholder for the weights
        self.w0 = tf.placeholder(
            shape=[None, ],
            dtype=self.dtype,
            name="w0"
        )

        # Drop placeholder
        self.should_drop = tf.placeholder(tf.bool, name="drop")

        # Regularization
        regularizer = tf.contrib.layers.l2_regularizer(self.weight_regularization)

        # Input_Dropout
        in0 = tf.layers.dropout(inputs=self.x0,
                                rate=self.input_dropout,
                                training=self.should_drop
                                )

        in1 = tf.layers.dropout(inputs=self.x1,
                                rate=self.input_dropout,
                                training=self.should_drop
                                )

        # Constructing the feature creation part of the net
        nn0 = tf.layers.dense(
            inputs=in0,
            units=self.hidden_layers[0],
            activation=self.feature_activation,
            use_bias=self.feature_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=regularizer,
            name="nn_hidden_0"
        )

        # By giving nn1 the same name as nn0 and using the flag reuse=True, 
        # the weights and biases of all neurons in each branch are identical
        nn1 = tf.layers.dense(
            inputs=in1,
            units=self.hidden_layers[0],
            activation=self.feature_activation,
            use_bias=self.feature_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=regularizer,
            name="nn_hidden_0",
            reuse=True
        )

        # Layer Dropout
        nn0 = tf.layers.dropout(inputs=nn0,
                                rate=self.dropout,
                                training=self.should_drop
                                )
        nn1 = tf.layers.dropout(inputs=nn1,
                                rate=self.dropout,
                                training=self.should_drop
                                )

        for i in range(1, len(self.hidden_layers)):
            nn0 = tf.layers.dense(
                inputs=nn0,
                units=self.hidden_layers[i],
                activation=self.feature_activation,
                use_bias=self.feature_bias,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=regularizer,
                name="nn_hidden_" + str(i)
            )
            nn1 = tf.layers.dense(
                inputs=nn1,
                units=self.hidden_layers[i],
                activation=self.feature_activation,
                use_bias=self.feature_bias,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=regularizer,
                name="nn_hidden_" + str(i),
                reuse=True
            )

            # Layer Dropout
            nn0 = tf.layers.dropout(inputs=nn0,
                                    rate=self.dropout,
                                    training=self.should_drop
                                    )
            nn1 = tf.layers.dropout(inputs=nn1,
                                    rate=self.dropout,
                                    training=self.should_drop
                                    )

        # Creating antisymmetric features for the ranking
        self.nn = (nn0 - nn1) / 2.

        self.nn = tf.layers.dense(
            inputs=self.nn,
            units=1,
            activation=self.ranking_activation,
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=regularizer,
            name="nn_rank"
        )

        self.nn_cls = tf.layers.dense(
            inputs=nn0 / 2.,
            units=1,
            activation=self.ranking_activation,
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=regularizer,
            name="nn_rank",
            reuse=True
        )

        nn_out = tf.identity(
            input=self.nn,
            name="nn"
        )

    def _build_pairs(self, query, samples, use_weights):
        """
        :param query: query of documents
        :param samples: number of samples
        :param use_weights: list of weights per document
        :return: a list of pairs of documents from a query
        """
        x0 = []
        x1 = []
        y = []
        if use_weights:
            w = []

        keys = sorted(list(query.keys()))
        for i in range(len(keys) - 1):
            indices0 = np.random.randint(0, len(query[keys[i + 1]]), samples)
            indices1 = np.random.randint(0, len(query[keys[i]]), samples)
            x0.extend(query[keys[i + 1]][indices0][:, :self.num_features])
            x1.extend(query[keys[i]][indices1][:, :self.num_features])
            y.extend((keys[i + 1] - keys[i]) * np.ones(samples))
            if use_weights:
                w.extend(self.weight_func(query[keys[i + 1]][indices0][-1],
                                          query[keys[i]][indices1][-1]))

        x0 = np.array(x0)
        x1 = np.array(x1)
        y = np.array([y]).transpose()
        if use_weights:
            w = np.array(w)
            return [x0, x1, y, w]
        else:
            return [x0, x1, y]

    def _build_pairs_f(self, query, targets, samples, use_weights):
        """
        :param query: query of documents
        :param targets: target values
        :param samples: number of samples
        :param use_weights: list of weights per document
        :return: a list of pairs of documents from a query
        """
        x0 = []
        x1 = []
        y = []
        if use_weights:
            w = []
        keys, counts = np.unique(targets, return_counts=True)
        sort_ids = np.argsort(keys)
        keys = keys[sort_ids]
        counts = counts[sort_ids]
        for i in range(len(keys) - 1):
            indices0 = np.random.randint(0, counts[i + 1], samples)
            indices1 = np.random.randint(0, counts[i], samples)
            querys0 = np.where(targets == keys[i + 1])[0]
            querys1 = np.where(targets == keys[i])[0]
            x0.extend(query[querys0][indices0][:, :self.num_features])
            x1.extend(query[querys1][indices1][:, :self.num_features])
            y.extend((keys[i + 1] - keys[i]) * np.ones(samples))
            if use_weights:
                w.extend(self.weight_func(query[querys0][indices0][-1],
                                          query[querys1][indices1][-1]))

        x0 = np.array(x0)
        x1 = np.array(x1)
        y = np.array([y]).transpose()
        if use_weights:
            w = np.array(w)
            return [x0, x1, y, w]
        else:
            return [x0, x1, y]

    def _build_no_query_pairs(self, features, samples, weights):
        """
        :param features: array of features
        :param samples: number of samples
        :param weights: list of weights per document
        :return: a list of pairs of instances
        """
        x0 = []
        x1 = []
        y = []
        if weights is not None:
            w = []

        keys = sorted(list(features.keys()))
        for i in range(len(keys) - 1):
            indices0 = np.random.randint(0, len(features[keys[i + 1]]), samples)
            indices1 = np.random.randint(0, len(features[keys[i]]), samples)
            x0.extend(features[keys[i + 1]][indices0])
            x1.extend(features[keys[i]][indices1])
            y.extend(keys[i + 1] * np.ones(samples))

        x0 = np.array(x0)
        x1 = np.array(x1)
        y = np.array([y]).transpose()
        if weights is None:
            return [x0, x1, y]
        else:
            w = np.array(w)
        return [x0, x1, y, w]

    def _fit_querys(self, dictOfQueries, validation, use_weights):
        """
        :param dictOfQueries: dict of queries for training the net. The key is the class
                            and the value is a list of queries
        :param use_weights: list of weights per document inside a query
        :return:
        """
        if self.x0 is None:
            if self.feature_func is None:
                self.num_features = len(dictOfQueries[list(dictOfQueries[0].keys())[0]][0][0]) - (
                    1 if use_weights else 0)
            else:
                if use_weights:
                    self.num_features = len(
                        self.feature_func(dictOfQueries[0][list(dictOfQueries.keys())[0]][0][0][:-1]))
                else:
                    self.num_features = len(self.feature_func(dictOfQueries[0][list(
                        dictOfQueries.keys())[0]][0][0]))

            self._build_model()

        if self.cost is None:
            if not self.weights:
                cost = tf.reduce_mean((self.y0 - self.nn) ** 2)
            else:
                cost = tf.reduce_mean(self.w0 * (self.y0 - self.nn) ** 2)
        else:
            cost = self.cost(self.nn, self.y0)

        # Regularization Loss
        l2_loss = tf.losses.get_regularization_loss()
        train_loss = cost + l2_loss

        global_step = tf.Variable(0, trainable=False)
        increment_global_step = tf.assign(global_step, global_step + 1)
        learning_rate = tf.train.exponential_decay(self.learning_rate,
                                                   global_step,
                                                   self.learning_rate_step_size,
                                                   self.learning_rate_decay_factor,
                                                   staircase=True)
        optimizer = self.optimizer(learning_rate).minimize(train_loss, global_step=global_step)
        init = tf.global_variables_initializer()
        sample_factor = np.log(1.0 * self.end_batch_size / self.start_batch_size)
        q_factor = np.log(1.0 * self.end_qids / self.start_qids)

        self.sess = tf.Session()
        self.sess.run(init)

        # Early Stopping
        scores = []
        best_sf = 0
        saver = tf.train.Saver()

        for step in range(self.max_steps):
            samples = int(self.start_batch_size * np.exp(1.0 * sample_factor * step / self.max_steps))
            q_samples = int(self.start_qids * np.exp(1.0 * q_factor * step / self.max_steps))

            x0 = []
            x1 = []
            y = []
            if use_weights:
                w = []

            queries = np.random.choice(dictOfQueries, q_samples)
            for q in queries:
                pairs = self._build_pairs(q, samples, use_weights)
                x0.extend(pairs[0])
                x1.extend(pairs[1])
                y.extend(pairs[2])
                if use_weights:
                    w.extend(pairs[3])

            if use_weights:
                val, _, _ = self.sess.run(
                    [cost, optimizer, increment_global_step],
                    feed_dict={self.x0: x0, self.x1: x1, self.w0: w, self.y0: y, self.should_drop: True})
            else:
                val, _, _ = self.sess.run(
                    [cost, optimizer, increment_global_step],
                    feed_dict={self.x0: x0, self.x1: x1, self.y0: y, self.should_drop: True})
            if self.print_step != 0 and step % self.print_step == 0:
                print("step: {}, value: {}, samples: {}, queries: {}".format(
                    step, val, samples, q_samples))

            # Early Stopping
            if self.early_stopping and step >= self.stop_start:
                cur_score = 0.
                for X, y, z in validation:
                    cur_score += self.stop_scorer(self, X, y)
                cur_score /= len(validation)
                scores.append(cur_score)

                if cur_score >= scores[best_sf] + self.stop_delta or step == self.stop_start:
                    best_sf = step - self.stop_start
                    saver.save(self.sess, "./tmp/{}_{}.ckpt".format(self.name, tmp_name))

                if step - best_sf > self.lookback:
                    saver.restore(self.sess, "./tmp/{}_{}.ckpt".format(self.name, tmp_name))
                    break

    def _fit_querys_f(self, dictOfQueries, validation, use_weights):
        """
        :param dictOfQueries: dict of queries for training the net. The key is the class
                            and the value is a list of queries
        :param use_weights: list of weights per document inside a query
        :return:
        """
        if self.x0 is None:
            if self.feature_func is None:
                len(dictOfQueries[0][0][0])
                self.num_features = len(dictOfQueries[0][0][0]) - (
                    1 if use_weights else 0)
            else:
                if use_weights:
                    self.num_features = len(
                        self.feature_func(dictOfQueries[0][0][0][:-1]))
                else:
                    self.num_features = len(self.feature_func(dictOfQueries[0][0][0]))

            self._build_model()

        if self.cost is None:
            if not self.weights:
                cost = tf.reduce_mean((self.y0 - self.nn) ** 2)
            else:
                cost = tf.reduce_mean(self.w0 * (self.y0 - self.nn) ** 2)
        else:
            cost = self.cost(self.nn, self.y0)

        # Regularization Loss
        l2_loss = tf.losses.get_regularization_loss()
        train_loss = cost + l2_loss

        global_step = tf.Variable(0, trainable=False)
        increment_global_step = tf.assign(global_step, global_step + 1)
        learning_rate = tf.train.exponential_decay(self.learning_rate,
                                                   global_step,
                                                   self.learning_rate_step_size,
                                                   self.learning_rate_decay_factor,
                                                   staircase=True)
        optimizer = self.optimizer(learning_rate).minimize(train_loss, global_step=global_step)
        init = tf.global_variables_initializer()
        sample_factor = np.log(1.0 * self.end_batch_size / self.start_batch_size)
        q_factor = np.log(1.0 * self.end_qids / self.start_qids)

        self.sess = tf.Session()
        self.sess.run(init)

        # Early Stopping
        tmp_name = str(time.time())
        scores = []
        best_sf = 0
        saver = tf.train.Saver()

        for step in range(self.max_steps):
            samples = int(self.start_batch_size * np.exp(1.0 * sample_factor * step / self.max_steps))
            q_samples = int(self.start_qids * np.exp(1.0 * q_factor * step / self.max_steps))

            x0 = []
            x1 = []
            y = []
            if use_weights:
                w = []

            queries = np.random.choice(len(dictOfQueries), q_samples)
            queries = [dictOfQueries[loc] for loc in queries]
            for X, y_2, z in queries:
                pairs = self._build_pairs_f(X, y_2, samples, use_weights)
                x0.extend(pairs[0])
                x1.extend(pairs[1])
                y.extend(pairs[2])
                if use_weights:
                    w.extend(pairs[3])

            if use_weights:
                val, _, _ = self.sess.run(
                    [cost, optimizer, increment_global_step],
                    feed_dict={self.x0: x0, self.x1: x1, self.w0: w, self.y0: y, self.should_drop: True})
            else:
                val, _, _ = self.sess.run(
                    [cost, optimizer, increment_global_step],
                    feed_dict={self.x0: x0, self.x1: x1, self.y0: y, self.should_drop: True})
            if self.print_step != 0 and step % self.print_step == 0:
                print("step: {}, value: {}, samples: {}, queries: {}".format(
                    step, val, samples, q_samples))

            # Early Stopping
            if self.early_stopping and step >= self.stop_start:
                cur_score = 0.
                for X, y, z in validation:
                    cur_score += self.stop_scorer(self, X, y)
                cur_score /= len(validation)
                scores.append(cur_score)

                if cur_score >= scores[best_sf] + self.stop_delta or step == self.stop_start:
                    best_sf = step - self.stop_start
                    saver.save(self.sess, "./tmp/{}_{}.ckpt".format(self.name, tmp_name))

                if step - best_sf > self.lookback:
                    saver.restore(self.sess, "./tmp/{}_{}.ckpt".format(self.name, tmp_name))
                    break

    def _fit_no_querys(self, features, validation, weights=None):
        """
                # ToDo for now refit a loaded ranker is not working
                :param features:
                :param real_classes:
                :param weights:
                :return:
                """
        if self.x0 is None:
            if self.feature_func is None:
                self.num_features = len(features[list(features.keys())[0]][0])
            else:
                self.num_features = len(self.feature_func(features[0]))
            self._build_model()

        if self.cost is None:
            if not self.weights:
                cost = tf.reduce_mean((self.y0 - self.nn) ** 2)
            else:
                cost = tf.reduce_mean(self.w0 * (self.y0 - self.nn) ** 2)
        else:
            cost = self.cost(self.nn, self.y0)

        # Regularization Loss
        l2_loss = tf.losses.get_regularization_loss()
        train_loss = cost + l2_loss

        global_step = tf.Variable(0, trainable=False)
        increment_global_step = tf.assign(global_step, global_step + 1)
        learning_rate = tf.train.exponential_decay(self.learning_rate,
                                                   global_step,
                                                   self.learning_rate_step_size,
                                                   self.learning_rate_decay_factor,
                                                   staircase=True)
        optimizer = self.optimizer(learning_rate).minimize(train_loss, global_step=global_step)
        init = tf.global_variables_initializer()
        sample_factor = np.log(1.0 * self.end_batch_size / self.start_batch_size)

        self.sess = tf.Session()
        self.sess.run(init)

        # Early Stopping
        tmp_name = str(time.time())
        scores = []
        best_sf = 0
        saver = tf.train.Saver()

        for step in range(self.max_steps):
            samples = int(self.start_batch_size * np.exp(1.0 * sample_factor * step / self.max_steps))
            pairs = self._build_no_query_pairs(features, samples, weights is not None)
            x0 = pairs[0]
            x1 = pairs[1]
            y = pairs[2]
            if weights is not None:
                w = pairs[3]

            if weights is not None:
                val, _, _ = self.sess.run(
                    [cost, optimizer, increment_global_step],
                    feed_dict={self.x0: x0, self.x1: x1, self.w0: w, self.y0: y, self.should_drop: True})
            else:
                val, _, _ = self.sess.run(
                    [cost, optimizer, increment_global_step],
                    feed_dict={self.x0: x0, self.x1: x1, self.y0: y, self.should_drop: True})
            if self.print_step != 0 and step % self.print_step == 0:
                print("step: {}, value: {}, samples: {}".format(step, val, samples))

            # Early Stopping
            if self.early_stopping and step >= self.stop_start:
                cur_score = 0.
                for X, y, z in validation:
                    cur_score += self.stop_scorer(self, X, y)
                cur_score /= len(validation)
                scores.append(cur_score)

                if cur_score >= scores[best_sf] + self.stop_delta or step == self.stop_start:
                    best_sf = step - self.stop_start
                    saver.save(self.sess, "./tmp/{}_{}.ckpt".format(self.name, tmp_name))

                if step - best_sf > self.lookback:
                    saver.restore(self.sess, "./tmp/{}_{}.ckpt".format(self.name, tmp_name))
                    break

    def fit(self, features, real_classes, **fit_params):
        """
        :param features: list of queries for training the net
        :param real_classes: list of labels inside a query
        :param weights: list of weights per document inside a query
        :return:
        """
        if "sample_weights" in fit_params.keys():
            sample_weights = fit_params["sample_weights"]
        else:
            sample_weights = None

        if fit_params["ranking"]:
            vals = []
            val_queries = []
            if self.early_stopping:
                val_queries = np.random.choice(len(features),
                                               int(len(features) * self.validation_size),
                                               replace=False)
                for i in val_queries:
                    vals.append((features[i], real_classes[i], sample_weights[i]
                    if sample_weights is not None else None))
            feats = []
            for i in range(len(features)):
                if i in val_queries:
                    continue
                feats.append((features[i], real_classes[i], sample_weights[i]
                if sample_weights is not None else None))
            self._fit_querys_f(feats, vals, sample_weights is not None)
        else:
            vals = None
            if self.early_stopping:
                id_train, id_test = train_test_split(
                    np.arange(len(features)), test_size=self.validation_size,
                    random_state=self.random_seed, shuffle=True, stratify=real_classes)
                vals = [(features[id_test], real_classes[id_test], sample_weights[id_test]
                if sample_weights is not None else None)]
                features = features[id_train]
                real_classes = real_classes[id_train]
                if sample_weights is not None:
                    sample_weights = sample_weights[id_train]

            feats = self._to_dict(features, real_classes, sample_weights if sample_weights is not None else None)
            self._fit_no_querys(feats, vals, sample_weights)

    @staticmethod
    def save(estimator, path):
        """
        This saves a saved directRanker
        :param path: location for the directRanker
        :param path:
        :return:
        """
        saver = tf.train.Saver()
        if "/" not in path:
            path = "./" + path
        saver.save(estimator.sess, path + ".ckpt")

        save_dr = directRanker()
        for key in estimator.get_params():
            # ToDo: Need to be fixed to also restore the cost function
            if key == "cost":
                save_dr.__setattr__(key, None)
            else:
                save_dr.__setattr__(key, estimator.get_params()[key])

        with open(path + ".pkl", 'wb') as output:
            pickle.dump(save_dr, output, 0)

    @staticmethod
    def load_ranker(path):
        """
        This loads a saved directRanker
        :param path: location for the saved directRanker
        :return:
        """
        tf.reset_default_graph()
        graph = tf.Graph()
        sess = tf.Session(graph=graph)
        with open(path + ".pkl", 'rb') as input:
            dr = pickle.load(input)

        with graph.as_default():
            saver = tf.train.import_meta_graph(path + ".ckpt.meta")
            saver.restore(sess, path + ".ckpt")
            dr.x0 = graph.get_tensor_by_name("x0:0")
            dr.x1 = graph.get_tensor_by_name("x1:0")
            dr.y0 = graph.get_tensor_by_name("y0:0")
            dr.w0 = graph.get_tensor_by_name("w0:0")
            dr.nn = graph.get_tensor_by_name("nn:0")
            dr.should_drop = graph.get_tensor_by_name("drop:0")
        dr.sess = sess
        dr.num_features = dr.x0.shape[1].value

        return dr

    def evaluate(self, features0, features1):
        """
        :param features0: list of features of the first instance feed to the net
        :param features1: list of features of the second instance feed to the net
        :return: r(features0, features1) of the net
        """
        if self.feature_func is None:
            features0 = np.array(features0)
            features1 = np.array(features1)
        else:
            features0, features1 = self.feature_func(features0, features1)
        if len(features0.shape) == 1:
            features0 = [features0]
            features1 = [features1]
        return self.sess.run(self.nn, feed_dict={self.x0: features0, self.x1: features1, self.should_drop: False})

    def evaluatePartNet(self, features):
        """
        :param features: list of features of the instance feed to the net
        :return: output of nn0/nn1 of the net
        """
        if self.feature_func_nn0_1 is None:
            features = np.array(features)
        else:
            features = self.feature_func_nn0_1(features)
        if len(features.shape) == 1:
            features = [features]

        return self.sess.run(self.nn_cls, feed_dict={self.x0: features, self.should_drop: False})

    def predict_proba(self, features):
        """
        :param features: list of features of the instance feed to the net
        :return: predicted class
        """
        if self.feature_func_nn0_1 is None:
            features = np.array(features)
        else:
            features = self.feature_func_nn0_1(features)
        if len(features.shape) == 1:
            features = [features]

        res = self.sess.run(self.nn_cls, feed_dict={self.x0: features, self.should_drop: False})

        return [0.5 * (value + 1) for value in res]

    def close(self):
        """
        This function closes the tensorflow session used for the directRanker
        """
        self.sess.close()
