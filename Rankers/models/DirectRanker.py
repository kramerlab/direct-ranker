import numpy as np
import tensorflow as tf

from sklearn.base import BaseEstimator
import gc

class DirectRanker(BaseEstimator):
    """
    TODO
    """

    def __init__(self,
                 # DirectRanker HPs
                 hidden_layers_dr=[256, 128, 64, 20],
                 feature_activation_dr='tanh',
                 ranking_activation_dr='sigmoid',
                 feature_bias_dr=True,
                 kernel_initializer_dr=tf.random_normal_initializer,
                 kernel_regularizer_dr=0.0,
                 drop_out=0,
                 # Common HPs
                 scale_factor_train_sample=1,
                 batch_size=200,
                 loss=tf.keras.losses.MeanSquaredError(),# 'binary_crossentropy'
                 learning_rate=0.01,
                 learning_rate_decay_rate=1,
                 learning_rate_decay_steps=1000,
                 optimizer=tf.keras.optimizers.Adam,# 'Nadam' 'SGD'
                 epoch=10,
                 steps_per_epoch=None,
                 # other variables
                 verbose=0,
                 validation_size=0.0,
                 num_features=0,
                 name="DirectRanker",
                 dtype=tf.float32,
                 print_summary=False,
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
        self.steps_per_epoch = steps_per_epoch
        # other variables
        self.verbose = verbose
        self.validation_size = validation_size
        self.num_features = num_features
        self.name = name
        self.dtype = dtype
        self.print_summary = print_summary

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
            activation=self.ranking_activation_dr,
            use_bias=False,
            kernel_regularizer=tf.keras.regularizers.l2(self.kernel_regularizer_dr),
            activity_regularizer=tf.keras.regularizers.l2(self.kernel_regularizer_dr),
            name="ranking_part"
        )(subtracted)

        if self.name == "RankNet" and self.ranking_activation_dr == "tanh":
            out = (out + 1) / 2.

        self.model = tf.keras.models.Model(
            inputs=[self.x0, self.x1],
            outputs=out,
            name='Stacker'
        )

        if self.learning_rate_decay_steps > 0:
            lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
                self.learning_rate,
                decay_steps=self.learning_rate_decay_steps,
                decay_rate=self.learning_rate_decay_rate,
                staircase=False
            )
        else:
            lr_schedule = self.learning_rate

        self.model.compile(
            optimizer=self.optimizer(lr_schedule),
            loss=self.loss,
            metrics=['acc']
        )

        if self.print_summary:
            self.model.summary()

    def fit(self, x, y, **fit_params):
        """
        TODO
        """
        self._build_model()

        x_l = []
        for c in np.unique(y):
            x_l.append(x[np.where(y == c)])

        for i in range(self.epoch):

            x0_cur = []
            x1_cur = []
            y_len = 0
            for c in np.unique(y):
                if c == max(np.unique(y)): continue
                idx0 = np.random.randint(0, len(x_l[c+1]), self.scale_factor_train_sample*len(x))
                idx1 = np.random.randint(0, len(x_l[c]), self.scale_factor_train_sample*len(x))
                y_len += len(idx0)
                x0_cur.extend(x_l[c+1][idx0])
                x1_cur.extend(x_l[c][idx1])
            x0_cur = np.array(x0_cur)
            x1_cur = np.array(x1_cur)

            if self.verbose > 0:
                print('Epoch {}/{}'.format(i + 1, self.epoch))

            self.model.fit(
                x=[x0_cur, x1_cur],
                y=np.ones(y_len),
                batch_size=self.batch_size,
                steps_per_epoch=self.steps_per_epoch,
                epochs=1,
                verbose=self.verbose,
                shuffle=True,
                validation_split=self.validation_size
            )
            # https://github.com/tensorflow/tensorflow/issues/14181
            # https://github.com/tensorflow/tensorflow/issues/30324
            gc.collect()

    def predict_proba(self, features):
        """
        TODO
        """
        if len(features.shape) == 1:
            features = [features]

        res = self.model.predict([features, np.zeros(np.shape(features))], batch_size=self.batch_size, verbose=self.verbose)

        return res

    def predict(self, features, threshold):
        """
        TODO
        """
        if len(features.shape) == 1:
            features = [features]

        features_conv = np.expand_dims(features, axis=2)

        res = self.model.predict([features, np.zeros(np.shape(features))], batch_size=self.batch_size, verbose=self.verbose)

        return [1 if r > threshold else 0 for r in res]
