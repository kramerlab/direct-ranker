import tensorflow as tf

from Rankers.models.DirectRanker import DirectRanker

def _ranknet_cost(y_actual, y_predicted):
    # mean(log(1+exp((1+pred)/2)) - (1+pred)/2)
    return tf.reduce_mean(tf.math.log(1+tf.math.exp((1+y_predicted)/2))-(1+y_predicted)/2)

class RankNet(DirectRanker):
    """
    TODO
    """

    def __init__(self,
        # DirectRanker HPs
        hidden_layers_dr=[256, 128, 64, 20],
        feature_activation_dr='relu',
        ranking_activation_dr='sigmoid',
        feature_bias_dr=True,
        kernel_initializer_dr=tf.random_normal_initializer,
        kernel_regularizer_dr=0.0,
        drop_out=0,
        # Common HPs
        scale_factor_train_sample=1,
        batch_size=200,
        loss=_ranknet_cost,
        learning_rate=0.0005,
        learning_rate_decay_rate=0,
        learning_rate_decay_steps=0,
        optimizer=tf.keras.optimizers.SGD,
        epoch=10,
        steps_per_epoch=None,
        # other variables
        verbose=0,
        validation_size=0.0,
        num_features=0,
        name="RankNet",
        dtype=tf.float32,
        print_summary=False,
    ):
        super().__init__(
            # DirectRanker HPs
            hidden_layers_dr=hidden_layers_dr,
            feature_activation_dr=feature_activation_dr,
            ranking_activation_dr=ranking_activation_dr,
            feature_bias_dr=feature_bias_dr,
            kernel_initializer_dr=kernel_initializer_dr,
            kernel_regularizer_dr=kernel_regularizer_dr,
            drop_out=drop_out,
            # Common HPs
            scale_factor_train_sample=scale_factor_train_sample,
            batch_size=batch_size,
            learning_rate=learning_rate,
            learning_rate_decay_rate=learning_rate_decay_rate,
            learning_rate_decay_steps=learning_rate_decay_steps,
            optimizer=optimizer,
            epoch=epoch,
            loss=loss,
            steps_per_epoch=steps_per_epoch,
            # other variables
            verbose=verbose,
            validation_size=validation_size,
            num_features=num_features,
            name=name,
            dtype=dtype,
            print_summary=print_summary,
        )
        
