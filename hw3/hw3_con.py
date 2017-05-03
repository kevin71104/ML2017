#############################################################
#                   Machine Learning 2017                   #
#            Hw3 : Image Sentiment Classification           #
#                Convolutional Neural Network               #
#    script : python3 train.csv loadmodel.h5 savemodel.h5   #
#  Description : use existed model and get further trained  #
#############################################################

import pandas as pd
import numpy as np
import sys
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator

######################### Read File #########################
with open(sys.argv[1],'r') as csvFile:
    train = pd.read_csv(csvFile)
train_feature = train['feature']
train_label   = np.array(train['label'])
train = []

x = []
for i in range(train_feature.shape[0]):
    x.append(train_feature[i].split(' '))
train_feature = np.array(x, dtype=float)
x =[]

train_feature = train_feature/255
classNum = 7

#validation Data
validNum = 5000
valid_label = train_label[ :validNum]
valid_feature = train_feature[ :validNum]
x_label = train_label[validNum:]
x_feature = train_feature[validNum: ]

train_label = []
train_feature = []

x_feature = x_feature.reshape(x_feature.shape[0],48,48,1)
valid_feature = valid_feature.reshape(valid_feature.shape[0],48,48,1)
x_label = np_utils.to_categorical(x_label, classNum)
valid_label = np_utils.to_categorical(valid_label, classNum)

######################### Start CNN #########################
model = load_model(sys.argv[2])

score = model.evaluate(x_feature, x_label)
print ("\nTrain accuracy:", score[1])
score2 = model.evaluate(valid_feature, valid_label)
print ("\nValid accuracy:", score2[1])

# Image Preprocessing - add noise
datagen = ImageDataGenerator(
    rotation_range=10.0,
    width_shift_range=0.1,
    height_shift_range=0.1
    )

batchNum = 100
for i in range(1):
    model.fit(x_feature, x_label,validation_data=(valid_feature,valid_label), batch_size = batchNum, epochs = 2)
    '''model.fit_generator(datagen.flow(x_feature, x_label, batch_size = batchNum),   # every flow has batchNum figures
                        steps_per_epoch = x_feature.shape[0]/batchNum,
                        epochs = 10,
                        validation_data = (valid_feature, valid_label)
                        )'''

model.save(sys.argv[3])

score = model.evaluate(x_feature, x_label)
print ("\nTrain accuracy:", score[1])
score2 = model.evaluate(valid_feature, valid_label)
print ("\nValid accuracy:", score2[1])
