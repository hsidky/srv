
import numpy as np 

from . import analysis 

import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.optimizers import Adam
import keras.layers as layers
from keras.regularizers import l2
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split


__all__ = ['HDE']

def minv(x, ret_sqrt=False):
    '''Utility function that returns the inverse of a matrix, with the
    option to return the square root of the inverse matrix.

    Parameters
    ----------
    x: numpy array with shape [m,m]
        matrix to be inverted
        
    ret_sqrt: bool, optional, default = False
        if True, the square root of the inverse matrix is returned instead

    Returns
    -------
    x_inv: numpy array with shape [m,m]
        inverse of the original matrix
    '''

    # Calculate eigvalues and eigvectors
    eigval_all, eigvec_all = tf.self_adjoint_eig(x)

    # Filter out eigvalues below threshold and corresponding eigvectors
    eig_th = tf.constant(K.epsilon(), dtype=tf.float32)
    index_eig = tf.to_int32(eigval_all > eig_th)
    _, eigval = tf.dynamic_partition(eigval_all, index_eig, 2)
    _, eigvec = tf.dynamic_partition(tf.transpose(eigvec_all), index_eig, 2)

    # Build the diagonal matrix with the filtered eigenvalues or square
    # root of the filtered eigenvalues according to the parameter
    eigval_inv = tf.diag(1/eigval)
    eigval_inv_sqrt = tf.diag(tf.sqrt(1/eigval))
    
    cond_sqrt = tf.convert_to_tensor(ret_sqrt)
    
    diag = tf.cond(cond_sqrt, lambda: eigval_inv_sqrt, lambda: eigval_inv)

    # Rebuild the square root of the inverse matrix
    x_inv = tf.matmul(tf.transpose(eigvec), tf.matmul(diag, eigvec))

    return x_inv



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


def create_orthogonal_encoder(encoder, input_size, n_components, means, gs_matrix, norms, order): 
    
    def layer(x, n_components=n_components, means=means, gs_matrix=gs_matrix, norms=norms, order=order):
        x -= means
        xs = []
        for i in range(n_components):
            xi = x[:,i]
            for j in range(i):
                xi -= gs_matrix[i, j]*xs[j]
            xs.append(xi)

        xo = K.stack(xs, axis=1)
        xo /= norms

        xo = K.stack([xo[:,i] for i in order], axis=1)
        return xo
    
    inp = layers.Input(shape=(input_size,))
    z = encoder(inp)
    z_orth = layers.Lambda(layer)(z)
    orth_encoder = Model(inp, z_orth)

    return orth_encoder


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
        self.autocorrelation_ = None
        self._sorted_idx = None
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
            self.encoder = create_orthogonal_encoder(
                            self._encoder, 
                            self.input_size, 
                            self.n_components,
                            self.empirical_means,
                            self.scaling_matrix, 
                            self.norm_factors,
                            self._sorted_idx
                        )


    @property
    def timescales_(self):
        if self.is_fitted:
            return -self.lag_time/np.log(self.autocorrelation_)
        
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
        lambdas = tf.Print(lambdas, [lambdas], summarize=100, first_n=10)
        #vamp_mat = K.dot(K.dot(minv(C00, ret_sqrt=True), C01), minv(C11, ret_sqrt=True))
        #vamp_mat = tf.Print(vamp_mat, [vamp_mat], summarize=100, first_n=5)
        #vamp_score = tf.norm(vamp_mat)
        return -1.0 - K.sum(lambdas**2)

    """
    def _loss(self, z_dummy, z):
        loss = 0
        zs = []
        for i in range(self.n_components):
            zi = z[:,i::self.n_components]
            zi -= K.mean(zi, axis=0)
            for zj in zs:
                zi -= K.mean(zi*zj, axis=0)/K.mean(zj*zj, axis=0)*zj
            
            zs.append(zi)
            loss += self.weights[i]/K.log(self._corr(zi[:,0], zi[:,1]))

        return loss
    """


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


    def _process_orthogonal_components(self, data, passive=False):
        if not passive:
            self.empirical_means = np.mean(data, axis=0)
        data -= self.empirical_means

        if not passive:
            self.scaling_matrix = np.ones((self.n_components, self.n_components))
        
        for i in range(self.n_components):
            for j in range(i):  
                if not passive:
                    gs_scale =  analysis.empirical_gram_schmidt(data[:,i], data[:,j])
                    self.scaling_matrix[i,j] = gs_scale
                    self.scaling_matrix[j,i] = gs_scale
                else:
                    gs_scale = self.scaling_matrix[i,j]
                
                data[:,i] -= gs_scale*data[:,j]
        
        if not passive:
            self.norm_factors = np.sqrt(np.mean(data*data, axis=0))
        
        return data

    def score(self, X, lag_time=None, score_k=None):
        if not self.is_fitted:
            raise RuntimeError('Model needs to be fit first.')

        if score_k is None:
            score_k = self.n_components
        
        x_t0, x_tt = self._create_dataset(X, lag_time=lag_time)
        z_t0 = self.transform(x_t0)
        z_tt = self.transform(x_tt)

        rho = np.array([analysis.empirical_correlation(z_t0[:,i], z_tt[:,i]) for i in range(score_k)])
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
        
        self.hde.fit(
            [train_x0, train_xt], 
            train_x0, 
            validation_data=[validation_data, validation_data[0]],
            callbacks=self.callbacks,
            batch_size=self.batch_size, 
            epochs=self.n_epochs, 
            verbose=self.verbose
        )
    
        if type(X) is list:
            temp = []
            for item in X:
                pred = self._encoder.predict(item, batch_size=self.batch_size)
                temp.append(pred)

            self._process_orthogonal_components(np.concatenate(temp))
            
            out = []
            for item in temp:
                out.append(self._process_orthogonal_components(item, passive=True))

        elif type(X) is np.ndarray:
            out = self._encoder.predict(X, batch_size=self.batch_size)
            out = self._process_orthogonal_components(out)
        else:
            raise TypeError('Data type {} is not supported'.format(type(X)))
        
        out_t0, out_tt = self._create_dataset(out)
        
        # Compute and store autocorrelation.
        self.autocorrelation_ = np.array([
            analysis.empirical_correlation(out_t0[:,i], out_tt[:,i]) 
            for i in range(self.n_components)])
        
        # Sort descending.
        self._sorted_idx = np.argsort(self.autocorrelation_)[::-1]
        self.autocorrelation_ = self.autocorrelation_[self._sorted_idx]

        self.encoder = create_orthogonal_encoder(
            self._encoder, 
            self.input_size, 
            self.n_components,
            self.empirical_means,
            self.scaling_matrix, 
            self.norm_factors,
            self._sorted_idx
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
