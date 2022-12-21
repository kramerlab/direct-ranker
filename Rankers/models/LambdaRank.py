import tensorflow as tf
from sklearn.base import BaseEstimator
import numpy as np
import gc
from ..helpers import ndcg_at_k


class LambdaRank(BaseEstimator):

    def __init__(self,
                 # DirectRanker HPs
                 hidden_layers_dr=[64, 20],
                 feature_activation_dr='sigmoid',
                 ranking_activation_dr='sigmoid',
                 feature_bias_dr=True,
                 kernel_initializer_dr=tf.random_normal_initializer,
                 kernel_regularizer_dr=0.0,
                 drop_out=0,
                 # Common HPs
                 scale_factor_train_sample=1,
                 batch_size=200,
                 loss=tf.keras.losses.BinaryCrossentropy,
                 learning_rate=0.001,
                 learning_rate_decay_rate=1,
                 learning_rate_decay_steps=1000,
                 optimizer=tf.keras.optimizers.Adam,  # 'Nadam' 'SGD'
                 epoch=10,
                 # other variables
                 verbose=0,
                 validation_size=0.0,
                 num_features=0,
                 name="DirectRanker",
                 dtype=tf.float32,
                 print_summary=False,
                 grad_clip=True,
                 clip_value=1.0,
                 sigma=1
                 ):

        # DirectRanker HPs
        self.hidden_layers_dr = hidden_layers_dr
        self.feature_activation_dr = feature_activation_dr
        self.ranking_activation_dr = ranking_activation_dr
        self.feature_bias_dr = feature_bias_dr
        self.kernel_initializer_dr = kernel_initializer_dr
        self.kernel_regularizer_dr = kernel_regularizer_dr
        self.drop_out = drop_out
        # Common HPs
        self.scale_factor_train_sample = scale_factor_train_sample
        self.batch_size = batch_size
        self.loss = loss
        self.learning_rate = learning_rate
        self.learning_rate_decay_rate = learning_rate_decay_rate
        self.learning_rate_decay_steps = learning_rate_decay_steps
        self.optimizer = optimizer
        self.epoch = epoch
        self.steps_per_epoch = None
        # other variables
        self.verbose = verbose
        self.validation_size = validation_size
        self.num_features = num_features
        self.name = name
        self.dtype = dtype
        self.print_summary = print_summary
        self.grad_clip = grad_clip
        self.sigma = sigma
        self.clip_value = clip_value

    def _build_model(self):
        """
        TODO
        """
        # Placeholders for the inputs
        self.x0 = tf.keras.layers.Input(
            shape=self.num_features,
            dtype=self.dtype,
            name="x0"
        )

        self.x1 = tf.keras.layers.Input(
            shape=self.num_features,
            dtype=self.dtype,
            name="x1"
        )

        input_layer = tf.keras.layers.Input(
            shape=self.num_features,
            dtype=self.dtype,
            name="input"
        )

        nn = tf.keras.layers.Dense(
            units=self.hidden_layers_dr[0],
            activation=self.feature_activation_dr,
            use_bias=self.feature_bias_dr,
            kernel_regularizer=tf.keras.regularizers.l2(self.kernel_regularizer_dr),
            bias_regularizer=tf.keras.regularizers.l2(self.kernel_regularizer_dr),
            activity_regularizer=tf.keras.regularizers.l2(self.kernel_regularizer_dr),
            name="nn_hidden_0"
        )(input_layer)

        if self.drop_out > 0:
            nn = tf.keras.layers.Dropout(self.drop_out)(nn)

        for i in range(1, len(self.hidden_layers_dr)):
            nn = tf.keras.layers.Dense(
                units=self.hidden_layers_dr[i],
                activation=self.feature_activation_dr,
                use_bias=self.feature_bias_dr,
                kernel_regularizer=tf.keras.regularizers.l2(self.kernel_regularizer_dr),
                bias_regularizer=tf.keras.regularizers.l2(self.kernel_regularizer_dr),
                activity_regularizer=tf.keras.regularizers.l2(self.kernel_regularizer_dr),
                name="nn_hidden_" + str(i)
            )(nn)

            if self.drop_out > 0:
                nn = tf.keras.layers.Dropout(self.drop_out)(nn)

        feature_part = tf.keras.models.Model(input_layer, nn, name='feature_part')

        if self.print_summary:
            feature_part.summary()

        nn0 = feature_part(self.x0)
        nn1 = feature_part(self.x1)

        subtracted = tf.keras.layers.Subtract()([nn0, nn1])

        out = tf.keras.layers.Dense(
            units=1,
            use_bias=False,
            kernel_regularizer=tf.keras.regularizers.l2(self.kernel_regularizer_dr),
            activity_regularizer=tf.keras.regularizers.l2(self.kernel_regularizer_dr),
            name="ranking_part",
            activation=tf.nn.sigmoid
        )(subtracted)

        self.model = tf.keras.models.Model(
            inputs=[self.x0, self.x1],
            outputs=out,
            name='LambdaRank'
        )

        lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
            self.learning_rate,
            decay_steps=self.learning_rate_decay_steps,
            decay_rate=self.learning_rate_decay_rate,
            staircase=False
        )

        self.optimizer = self.optimizer(lr_schedule)

        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss(),
            metrics=['acc']
        )

        if self.print_summary:
            self.model.summary()

    def fit(self, x, y, **fit_params):
        """
        TODO
        """
        tf.keras.backend.clear_session()
    

        self._build_model()
        self.steps_per_epoch = np.ceil(len(x) / self.batch_size)
        y = y.reshape(-1, 1)
        
        for i in range(self.epoch):   
            idx0 = np.random.randint(0, len(x), self.scale_factor_train_sample*len(x))
            idx1 = np.random.randint(0, len(x), self.scale_factor_train_sample*len(x))
            
            x0_cur = []
            x1_cur = [] 
            y_cur = []
            for i0, i1 in zip(idx0, idx1):
                if y[i0] - y[i1] != 0:
                    if y[i0] - y[i1] > 0: y_cur.append([1])
                    if y[i0] - y[i1] < 0: y_cur.append([-1])
                    x0_cur.append(x[i0])
                    x1_cur.append(x[i1])
            x0_cur = np.array(x0_cur).astype(np.float32)
            x1_cur = np.array(x1_cur).astype(np.float32)
            y_cur = np.array(y_cur).astype(np.float32)

            for j in range(int(self.steps_per_epoch)):
                if len(x0_cur) == 0: continue
                idx = np.random.randint(0, len(x0_cur), self.batch_size)
                step, train_loss, score = self._train_step((x0_cur[idx], x1_cur[idx]), y_cur[idx])
                if step % 100 == 0:
                    ndcg20, ndcg500 = self._get_ndcg(y, score)

    @tf.function
    def _train_step(self, x, y):
        with tf.GradientTape(persistent=True) as tape:
            #tape.watch(self.x0)
            #tape.watch(self.x1)
            pred_score = self.get_scores(x)
            loss = tf.reduce_mean(self._ranknet_loss(pred_score, y))
            lambdas = self._get_lambdas(pred_score, y)
            pred_score = tf.reshape(pred_score, [-1])

        with tf.name_scope("gradients"):
            gradients = [self._get_lambda_scaled_derivative(tape, pred_score, Wk, lambdas)
                         for Wk in self.model.trainable_variables]
            if self.grad_clip:
                gradients = [(tf.clip_by_value(grad, -self.clip_value, self.clip_value))
                             for grad in gradients]
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            step = self.optimizer.iterations

        return step, loss, pred_score

    def _ranknet_loss(self, pred_scores, y):
        return tf.keras.backend.binary_crossentropy(y, pred_scores)

    def _get_lambda_scaled_derivative(self, tape, pred_score, Wk, lambdas):
        dsi_dWk = tape.jacobian(pred_score, Wk)  # ∂si/∂wk
        dsi_dWk_minus_dsj_dWk = tf.expand_dims(dsi_dWk, 1) - tf.expand_dims(dsi_dWk, 0)  # ∂si/∂wk−∂sj/∂wk
        shape = tf.concat([tf.shape(lambdas),
                           tf.ones([tf.rank(dsi_dWk_minus_dsj_dWk) - tf.rank(lambdas)],
                                   dtype=tf.int32)], axis=0)
        # (1/2(1−Sij)−1/1+eσ(si−sj))(∂si/∂wk−∂sj/∂wk)
        grad = tf.reshape(lambdas, shape) * dsi_dWk_minus_dsj_dWk
        grad = tf.reduce_mean(grad, axis=[0, 1])
        return grad

    def _get_lambdas(self, pred_score, labels):
        """https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf
        As explained in equation 3
        (1/2(1-Sij)-1/1+e\sigma(si-sj))"""
        with tf.name_scope("lambdas"):
            batch_size = tf.shape(labels)[0]

            index = tf.reshape(tf.range(1.0, tf.cast(batch_size,
                                                     dtype=tf.float32) + 1),
                               tf.shape(labels))

            sorted_labels = tf.sort(labels,
                                    direction="DESCENDING",
                                    axis=0)

            diff_matrix = labels - tf.transpose(labels)
            label_diff_matrix = tf.maximum(tf.minimum(1., diff_matrix), -1.)
            pred_diff_matrix = pred_score - tf.transpose(pred_score)
            lambdas = self.sigma * ((1 / 2) * (1 - label_diff_matrix) -
                                    tf.nn.sigmoid(-self.sigma * pred_diff_matrix))

            with tf.name_scope("ndcg"):
                cg_discount = tf.math.log(1.0 + index)
                rel = 2 ** labels - 1
                sorted_rel = 2 ** sorted_labels - 1
                dcg_m = rel / cg_discount
                dcg = tf.reduce_sum(dcg_m)

                stale_ij = tf.tile(dcg_m, [1, batch_size])
                new_ij = rel / tf.transpose(cg_discount)
                stale_ji = tf.transpose(stale_ij)
                new_ji = tf.transpose(new_ij)
                dcg_new = dcg - stale_ij + new_ij - stale_ji + new_ji
                dcg_max = tf.reduce_sum(sorted_rel / cg_discount)
                ndcg_delta = tf.math.abs(dcg_new - dcg) / dcg_max


            lambdas = lambdas * ndcg_delta

        return lambdas

    def predict_proba(self, features):
        """
        TODO
        """
        if len(features.shape) == 1:
            features = [features]

        res = self.model.predict([features, np.zeros(np.shape(features))], batch_size=self.batch_size,
                                 verbose=self.verbose)

        return res

    def get_scores(self, x_pair):
        scores = tf.nn.sigmoid(self.model(x_pair, training=False))
        return scores

    def call(self, inputs, training=False, threshold=0.5):
        res = self.model(inputs)
        if training:
            return res
        else:
            res = self.model(inputs)
            return [1 if r > threshold else 0 for r in res]

    @staticmethod
    def _get_ndcg(target, pred_score):
        # print(tf.shape(pred_score))
        target = tf.reshape(target, [-1])

        # print(tf.shape(target))
        zpd = list(zip(target.numpy(), pred_score.numpy()))
        zpd.sort(key=lambda x: x[1], reverse=True)
        pred_rank, _ = list(zip(*zpd))

        test_ndcg_5 = ndcg_at_k(list(pred_rank), 5)
        test_ndcg_20 = ndcg_at_k(list(pred_rank), 20)

        return test_ndcg_5, test_ndcg_20
