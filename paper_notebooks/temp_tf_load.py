import tensorflow as tf
import keras.backend as K
from keras.models import load_model
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.tensorflow_backend.set_session(tf.Session(config=config))

