## Code based on WGAN implementation in https://github.com/eriklindernoren/Keras-GAN

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from keras.layers.merge import _Merge
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from functools import partial
import keras.backend as K

from keras.utils import get_custom_objects, plot_model


__all__ = ['MolGen']


def swish(x):
    return (K.sigmoid(x) * x)


class RandomWeightedAverage(_Merge):
    
    def __init__(self, batch_size):
        _Merge.__init__(self)
        self.batch_size=batch_size


    """Provides a (random) weighted average between real and generated image samples"""
    def _merge_function(self, inputs):
        alpha = K.random_uniform((self.batch_size, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])


class MolGen(BaseEstimator, TransformerMixin):

    def __init__(self, latent_dim, output_dim, batch_size=1000, noise_dim=50,
                n_epochs=1000,
                hidden_layer_depth=3, hidden_size=200, activation='swish', verbose=True,
                n_discriminator=5):
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.hidden_layer_depth = hidden_layer_depth
        self.hidden_size = hidden_size
        self.activation = activation
        self.batch_size=batch_size
        self.n_epochs = n_epochs
        self.noise_dim = noise_dim
        self.verbose = verbose

        self.n_discriminator = n_discriminator
        optimizer = RMSprop(lr=0.00005)

        # Register swish
        get_custom_objects().update({'swish': swish})

        # Build generator and discriminator 
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()

        #
        # Construct computational graph for discriminator
        #
        self.generator.trainable = False 

        # Real molecule 
        real_mol = Input(shape=(self.output_dim,))

        # Discriminator input 
        z_disc = Input(shape=(self.latent_dim + self.noise_dim,))
        fake_mol = self.generator(z_disc)

        # conditioning input 
        z_cond = Input(shape=(self.latent_dim,))
        # Condition both fake and valid. 
        fake_mol_cond = Concatenate(axis=1)([fake_mol, z_cond])
        real_mol_cond = Concatenate(axis=1)([real_mol, z_cond])

        # Discriminator does its job 
        fake = self.discriminator(fake_mol_cond)
        valid = self.discriminator(real_mol_cond)


        # Interpolated between real and fake molecule.
        interp_mol = RandomWeightedAverage(batch_size=self.batch_size)([real_mol_cond, fake_mol_cond])
        validity_interp = self.discriminator(interp_mol)

        # Use Python partial to provide loss function with additional
        # 'averaged_samples' argument
        partial_gp_loss = partial(self.ـgradient_penalty_loss,
                          averaged_samples=interp_mol)
        partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names

        self.discriminator_model = Model(inputs=[real_mol, z_disc, z_cond],
                            outputs=[valid, fake, validity_interp])
        self.discriminator_model.compile(loss=[self._wasserstein_loss,
                                        self._wasserstein_loss,
                                        partial_gp_loss],
                                        optimizer=optimizer,
                                        loss_weights=[1, 1, 10])

        # 
        # Construct computational graph for generator 
        # 
        self.discriminator.trainable = False
        self.generator.trainable = True

        z_gen = Input(shape=(self.latent_dim + self.noise_dim,))
        mol = self.generator(z_gen)
        z_cond = Input(shape=(self.latent_dim,))
        mol_cond = Concatenate(axis=1)([mol, z_cond])
        valid = self.discriminator(mol_cond)
        self.generator_model = Model([z_gen, z_cond], valid)
        self.generator_model.compile(loss=self._wasserstein_loss, optimizer=optimizer)

        self.is_fitted = False


    def fit(self, X, y): 
        # Adversarial ground truths
        valid = -np.ones((self.batch_size, 1))
        fake =  np.ones((self.batch_size, 1))
        dummy = np.zeros((self.batch_size, 1)) # Dummy gt for gradient penalty
        for epoch in range(self.n_epochs):
            for _ in range(self.n_discriminator):
                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                idx = np.random.randint(0, y.shape[0], self.batch_size)
                real_mols = y[idx]

                # Sample generator input
                fake_mols = np.concatenate([X[idx], np.random.normal(0, 1, (self.batch_size, self.noise_dim))], axis=1)
                
                # Train the critic
                d_loss = self.discriminator_model.train_on_batch([real_mols, fake_mols, X[idx]],
                                                            [valid, fake, dummy])

            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = self.generator_model.train_on_batch([fake_mols, X[idx]], valid)

            # Plot the progress
            if self.verbose:
                print ("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss))
        
        self.is_fitted = True
        return self


    def transform(self, X):

        if self.is_fitted:
            out = self.generator.predict(np.concatenate([X, np.random.normal(0, 1, (X.shape[0], self.noise_dim))], axis=1), batch_size=self.batch_size)
            return out
        
        raise RuntimeError('Model needs to be fit first.')


    def ـgradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        gradients = K.gradients(y_pred, averaged_samples)[0]
        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)


    def _wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)


    def _build_generator(self):
        model = Sequential()

        model.add(
            Dense(
                self.hidden_size, 
                activation=self.activation, 
                input_dim=self.latent_dim + self.noise_dim
            )
        )
        for _ in range(self.hidden_layer_depth - 1):
            model.add(Dense(self.hidden_size, activation=self.activation))

        model.add(Dense(self.output_dim, activation='tanh'))

        model.summary()

        input = Input(shape=(self.latent_dim + self.noise_dim,))
        mol = model(input)

        return Model(input, mol)

    def _build_discriminator(self):
        model = Sequential()

        model.add(
            Dense(
                self.hidden_size, 
                activation=self.activation, 
                input_dim=self.output_dim + self.latent_dim
            )
        )
        for _ in range(self.hidden_layer_depth - 1):
            model.add(Dense(self.hidden_size, activation=self.activation))
        
        model.add(Dense(1))

        model.summary()

        mol = Input(shape=(self.output_dim + self.latent_dim,))
        validity = model(mol)

        return Model(mol, validity)


        
        



