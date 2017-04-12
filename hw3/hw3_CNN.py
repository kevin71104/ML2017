#############################################################
#                   Machine Learning 2017                   #
#            Hw3 : Image Sentiment Classification           #
#                Convolutional Neural Network               #
#############################################################

import pandas as pd
import numpy as np
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
