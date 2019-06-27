import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Lambda
from tensorflow.python.keras.layers import Layer
from tensorflow.keras.utils import get_custom_objects, plot_model
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions


__all__ = ['Propagator']


def swish(x):
    return (K.sigmoid(x) * x)


def elu_plus_one_plus_epsilon(x):
    """ELU activation with a very small addition to help prevent
    NaN in loss."""
    return K.elu(x) + 1 + K.epsilon()


class MDN(Layer):
    """A Mixture Density Network Layer for Keras.
    This layer has a few tricks to avoid NaNs in the loss function when training:
        - Activation for variances is ELU + 1 + 1e-8 (to avoid very small values)
        - Mixture weights (pi) are trained in as logits, not in the softmax space.

    A loss function needs to be constructed with the same output dimension and number of mixtures.
    A sampling function is also provided to sample from distribution parametrised by the MDN outputs.
    """

    def __init__(self, output_dimension, num_mixtures, **kwargs):
        self.output_dim = output_dimension
        self.num_mix = num_mixtures

        with tf.name_scope('MDN'):
            self.mdn_mus = Dense(self.num_mix * self.output_dim, name='mdn_mus', activation='sigmoid')  # mix*output vals, no activation
            self.mdn_sigmas = Dense(self.num_mix * self.output_dim, activation=elu_plus_one_plus_epsilon, name='mdn_sigmas')  # mix*output vals exp activation
            self.mdn_pi = Dense(self.num_mix, name='mdn_pi', activation='softmax')  # mix vals, logits
        super(MDN, self).__init__(**kwargs)
        

    def build(self, input_shape):
        self.mdn_mus.build(input_shape)
        self.mdn_sigmas.build(input_shape)
        self.mdn_pi.build(input_shape)
        self._trainable_weights = self.mdn_mus.trainable_weights + self.mdn_sigmas.trainable_weights + self.mdn_pi.trainable_weights
        self._non_trainable_weights = self.mdn_mus.non_trainable_weights + self.mdn_sigmas.non_trainable_weights + self.mdn_pi.non_trainable_weights
        super(MDN, self).build(input_shape)


    def call(self, x, mask=None):
        with tf.name_scope('MDN'):
            mdn_out = keras.layers.concatenate([self.mdn_mus(x),
                                                self.mdn_sigmas(x),
                                                self.mdn_pi(x)],
                                               name='mdn_outputs')
        return mdn_out


    def compute_output_shape(self, input_shape):
        """Returns output shape, showing the number of mixture parameters."""
        return (input_shape[0], (2 * self.output_dim * self.num_mix) + self.num_mix)


    def get_config(self):
        config = {
            "output_dimension": self.output_dim,
            "num_mixtures": self.num_mix
        }
        base_config = super(MDN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def get_mixture_loss_func(output_dim, num_mixes):
    """Construct a loss functions for the MDN layer parametrised by number of mixtures."""
    # Construct a loss function with the right number of mixtures and outputs
    def loss_func(y_true, y_pred):
        # Reshape inputs in case this is used in a TimeDistribued layer
        y_pred = tf.reshape(y_pred, [-1, (2 * num_mixes * output_dim) + num_mixes], name='reshape_ypreds')
        y_true = tf.reshape(y_true, [-1, output_dim], name='reshape_ytrue')
        # Split the inputs into paramaters
        out_mu, out_sigma, out_pi = tf.split(y_pred, num_or_size_splits=[num_mixes * output_dim,
                                                                         num_mixes * output_dim,
                                                                         num_mixes],
                                             axis=-1, name='mdn_coef_split')
        
        mus = tf.reshape(out_mu, (-1, num_mixes, output_dim))
        sigs = tf.reshape(out_sigma, (-1, num_mixes, output_dim))
        
        cat = tfd.Categorical(probs=K.clip(out_pi, 1e-8, 1.))
        
        mixture = tfd.MixtureSameFamily(
            mixture_distribution=cat, 
            components_distribution=tfd.MultivariateNormalDiag(loc=mus, scale_diag=sigs) 
        )
        loss = mixture.log_prob(y_true)
        loss = tf.negative(loss)
        loss = tf.reduce_mean(loss)
        return loss

    # Actually return the loss_func
    with tf.name_scope('MDN'):
        return loss_func


def get_mixture_fun(output_dim, num_mixes):
    """Construct a TensorFlor sampling operation for the MDN layer parametrised
    by mixtures and output dimension. This can be used in a Keras model to
    generate samples directly."""

    def sampling_func(y_pred):
        # Reshape inputs in case this is used in a TimeDistribued layer
        y_pred = tf.reshape(y_pred, [-1, (2 * num_mixes * output_dim) + num_mixes], name='reshape_ypreds')
        out_mu, out_sigma, out_pi = tf.split(y_pred, num_or_size_splits=[num_mixes * output_dim,
                                                                         num_mixes * output_dim,
                                                                         num_mixes],
                                             axis=1, name='mdn_coef_split')
        

        mus = tf.reshape(out_mu, (-1, num_mixes, output_dim))
        sigs = tf.reshape(out_sigma, (-1, num_mixes, output_dim))
        
        cat = tfd.Categorical(probs=K.clip(out_pi, 1e-8, 1.))
        
        mixture = tfd.MixtureSameFamily(
            mixture_distribution=cat, 
            components_distribution=tfd.MultivariateNormalDiag(loc=mus, scale_diag=sigs) 
        )        

        return mixture

    # Actually return the loss_func
    with tf.name_scope('MDNLayer'):
        return sampling_func


class Propagator(BaseEstimator, TransformerMixin):

    def __init__(self, input_dim, n_components=2, lag_time=100, batch_size=1000, 
                learning_rate=0.001, n_epochs=100, hidden_layer_depth=2, 
                hidden_size=100, activation='swish', callbacks=None, verbose=True):
        self.input_dim = input_dim 
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.lag_time = lag_time
        self.callbacks = callbacks

        model = keras.Sequential()
        get_custom_objects().update({'swish': swish})
        
        model.add(Dense(hidden_size, activation=activation, input_shape=(input_dim,)))

        for _ in range(hidden_layer_depth - 1):
            model.add(Dense(hidden_size, activation=activation))
        
        model.add(MDN(input_dim, n_components))
        model.compile(loss=get_mixture_loss_func(input_dim, n_components), optimizer=keras.optimizers.Adam(lr=learning_rate))

        self.model = model
        self.is_fitted = False

    
    def fit(self, X, y=None): 
        x0, xt = self._create_dataset(X)
        
        self.model.fit(x0, xt, batch_size=self.batch_size, 
                    epochs=self.n_epochs, verbose=self.verbose,
                    callbacks=self.callbacks)
        
        self.is_fitted = True
        return self

    
    def transform(self, X):

        if self.is_fitted:
            out = self.model.predict(X, batch_size=self.batch_size)
            return out
        
        raise RuntimeError('Model needs to be fit first.')


    def _create_dataset(self, data, lag_time=None):
        if lag_time is None:
            lag_time = self.lag_time

        if type(data) is list: 
            x_t0 = []
            x_tt = []
            for item in data:
                x_t0.append(item[:-lag_time])
                x_tt.append(item[lag_time:])
            
            x_t0 = np.concatenate(x_t0)
            x_tt = np.concatenate(x_tt) 
        elif type(data) is np.ndarray:
            x_t0 = data[:-lag_time]
            x_tt = data[lag_time:]
        else:
            raise TypeError('Data type {} is not supported'.format(type(data)))

        return [x_t0, x_tt]
