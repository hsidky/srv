
import numpy as np

from . import analysis 

import tensorflow as tf
import scipy.linalg
from keras import backend as K
from keras.models import Model
from keras.optimizers import Adam
import keras.layers as layers
from keras.regularizers import l2
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split


__all__ = ['HDE']


def create_encoder(input_size, output_size, hidden_layer_depth, 
                   hidden_size, dropout_rate, noise_std, l2_reg, 
                   batch_norm, activation):
    encoder_input = layers.Input(shape=(input_size,))
    encoder = layers.Dense(
                        hidden_size, 
                        activation=activation, 
                        kernel_regularizer=l2(l2_reg)
                    )(encoder_input)
    for _ in range(hidden_layer_depth - 1):
        if batch_norm:
            encoder = layers.BatchNormalization(axis=1)(encoder)

        encoder = layers.Dense(
                            hidden_size, 
                            activation=activation,
                            kernel_regularizer=l2(l2_reg)
                        )(encoder)

        if dropout_rate > 0:
            encoder = layers.Dropout(dropout_rate)(encoder)
    
    encoder = layers.Dense(output_size, activation=activation)(encoder)
    encoder = layers.GaussianNoise(stddev=noise_std)(encoder)
    model = Model(encoder_input, encoder)
    return model


def create_hde(encoder, input_size):
    input_t0 = layers.Input(shape=(input_size,))
    input_tt = layers.Input(shape=(input_size,))
    z_t0 = encoder(input_t0)
    z_tt = encoder(input_tt)
    z = layers.Concatenate(axis=1)([z_t0, z_tt])
    hde = Model([input_t0, input_tt], z)
    return hde


def create_vac_encoder(encoder, input_size, n_components, means, eigenvectors, norms):
    k_means = K.variable(means)
    k_eigenvectors = K.variable(eigenvectors)
    k_norms = K.variable(norms)

    def layer(x, n_components=n_components, means=k_means, eigenvectors=k_eigenvectors, norms=k_norms):
        x -= means
        z = K.dot(x, eigenvectors)
        z /= norms
        return z
    
    inp = layers.Input(shape=(input_size,))
    z = encoder(inp)
    z_vac = layers.Lambda(layer)(z)
    vac_encoder = Model(inp, z_vac)

    return vac_encoder
        

class HDE(BaseEstimator, TransformerMixin):

    def __init__(self, input_size, n_components=2, lag_time=1, n_epochs=100, 
                 learning_rate=0.001, dropout_rate=0, l2_regularization=0., 
                 hidden_layer_depth=2, hidden_size=100, activation='tanh', 
                 batch_size=100, validation_split=0, callbacks=None, 
                 batch_normalization=False, latent_space_noise=0, verbose=True):

        self._encoder = create_encoder(input_size, n_components, hidden_layer_depth,
                                       hidden_size, dropout_rate, latent_space_noise, 
                                       l2_regularization, batch_normalization, 
                                       activation)
        self.encoder = self._encoder
        self.hde = create_hde(self._encoder, input_size)

        self.input_size = input_size
        self.n_components = n_components
        self.lag_time = lag_time
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.l2_regularization = l2_regularization
        self.batch_size = batch_size
        self.verbose = verbose
        self.validation_split = validation_split
        self.callbacks = callbacks
        self.batch_normalization = batch_normalization
        self.latent_space_noise = latent_space_noise

        self.weights = np.ones(self.n_components)

        # Cached variables 
        self.eigenvalues_ = None
        self.eigenvectors_ = None
        self.means_ = None 
        self._recompile = False

        self.is_fitted = False


    def __getstate__(self):
        d = self.__dict__.copy()
        d.pop('encoder')
        d.pop('hde')
        d.pop('optimizer')
        return d


    def __setstate__(self, state):
        self.__dict__.update(state)
        self.learning_rate = self._learning_rate_
        self.hde = create_hde(self._encoder, self.input_size)
        if self.is_fitted:
            self.encoder = create_vac_encoder(
                self._encoder,
                self.input_size,
                self.n_components,
                self.means_,
                self.eigenvectors_,
                self.norms_
            )


    @property
    def timescales_(self):
        if self.is_fitted:
            return -self.lag_time/np.log(self.eigenvalues_)
        
        raise RuntimeError('Model needs to be fit first.')


    @property
    def learning_rate(self):
        return self._learning_rate_
    

    @learning_rate.setter
    def learning_rate(self, value):
        self._learning_rate_ = value
        self.optimizer = Adam(lr=value)
        self._recompile = True


    def _corr(self, x, y):
        xc = x - K.mean(x)
        yc = y - K.mean(y)
        corr = K.mean(xc*yc)/(K.std(x)*K.std(y))
        return corr


    def _loss(self, z_dummy, z):
        N = tf.to_float(tf.shape(z)[0])

        z_t0 = z[:, :self.n_components]
        z_t0 -= K.mean(z_t0, axis=0)
        
        z_tt = z[:, self.n_components:]
        z_tt -= K.mean(z_tt, axis=0)

        C00 = 1/(N - 1)*K.dot(K.transpose(z_t0), z_t0)
        C01 = 1/(N - 1)*K.dot(K.transpose(z_t0), z_tt)
        C10 = 1/(N - 1)*K.dot(K.transpose(z_tt), z_t0)
        C11 = 1/(N - 1)*K.dot(K.transpose(z_tt), z_tt)

        C0 = 0.5*(C00 + C11)
        C1 = 0.5*(C01 + C10)
        
        L = tf.cholesky(C0)
        Linv = tf.matrix_inverse(L)

        A = K.dot(K.dot(Linv, C1), K.transpose(Linv))

        lambdas, vs = tf.self_adjoint_eig(A)
        return -1.0 - K.sum(lambdas**2)


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


    def _calculate_basis(self, x_t0, x_tt):
        N = x_t0.shape[0]
        x = np.concatenate([x_t0, x_tt])
        self.means_ = np.mean(x, axis=0)

        x_t0m = x_t0 - self.means_
        x_ttm = x_tt - self.means_

        C00 = 1/(N - 1)*x_t0m.T.dot(x_t0m)
        C01 = 1/(N - 1)*x_t0m.T.dot(x_ttm)
        C10 = 1/(N - 1)*x_ttm.T.dot(x_t0m)
        C11 = 1/(N - 1)*x_ttm.T.dot(x_ttm)

        C0 = 0.5*(C00 + C11)
        C1 = 0.5*(C01 + C10)

        eigvals, eigvecs = scipy.linalg.eigh(C1, b=C0)
        idx = np.argsort(eigvals)[::-1]

        self.eigenvalues_ = eigvals[idx]
        self.eigenvectors_ = eigvecs[:, idx]
        self.norms_ = np.sqrt(np.mean(x*x, axis=0))


    def score(self, X, lag_time=None, score_k=None):
        if not self.is_fitted:
            raise RuntimeError('Model needs to be fit first.')

        if score_k is None:
            score_k = self.n_components + 1
        
        x_t0, x_tt = self._create_dataset(X, lag_time=lag_time)
        z_t0 = self.transform(x_t0)
        z_tt = self.transform(x_tt)

        rho = np.array([analysis.empirical_correlation(z_t0[:,i], z_tt[:,i]) for i in range(score_k - 1)])
        score = 1. + np.sum(rho**2)

        return score


    def fit(self, X, y=None):
        all_data = self._create_dataset(X)
            
        if y is not None:
            train_x0, train_xt = all_data
            validation_data = self._create_dataset(y)
        else:
            train_x0, val_x0, train_xt, val_xt = train_test_split(all_data[0], all_data[1], test_size=self.validation_split)
            validation_data = [val_x0, val_xt]
        
        if not self.is_fitted or self._recompile:
            self.hde.compile(optimizer=self.optimizer, loss=self._loss)
            self._recompile = False
        
        self.history = self.hde.fit(
            [train_x0, train_xt], 
            train_x0, 
            validation_data=[validation_data, validation_data[0]],
            callbacks=self.callbacks,
            batch_size=self.batch_size, 
            epochs=self.n_epochs, 
            verbose=self.verbose
        )
    
        if type(X) is list:
            out = [self._encoder.predict(x, batch_size=self.batch_size) for x in X]
        elif type(X) is np.ndarray:
            out = self._encoder.predict(X, batch_size=self.batch_size)
        else:
            raise TypeError('Data type {} is not supported'.format(type(X)))
        
        out_t0, out_tt = self._create_dataset(out)
        
        self._calculate_basis(out_t0, out_tt)

        self.encoder = create_vac_encoder(
            self._encoder,
            self.input_size,
            self.n_components,
            self.means_,
            self.eigenvectors_,
            self.norms_
        )

        self.is_fitted = True
        return self


    def transform(self, X):

        if self.is_fitted:
            out = self.encoder.predict(X, batch_size=self.batch_size)
            return out
        
        raise RuntimeError('Model needs to be fit first.')


    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
