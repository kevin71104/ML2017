#############################################################
#                   Machine Learning 2017                   #
#            Hw3 : Image Sentiment Classification           #
#                Convolutional Neural Network               #
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
#train_feature = train['feature'].str.split(' ')
#train_feature = train_feature.tolist()
train_feature = train['feature']
x = []
for i in range(train_feature.shape[0]):
    x.append(train_feature[i].split(' '))
train_feature = np.array(x, dtype=float)
train_feature = train_feature/255

train_label   = np.array(train['label'])
classNum = 7

#validation Data
validNum = 5700
valid_label = train_label[ :validNum]
valid_feature = train_feature[ :validNum]
x_label = train_label[validNum:]
x_feature = train_feature[validNum: ]

x_feature = x_feature.reshape(x_feature.shape[0],48,48,1)
valid_feature = valid_feature.reshape(valid_feature.shape[0],48,48,1)
x_label = np_utils.to_categorical(x_label, classNum)
valid_label = np_utils.to_categorical(valid_label, classNum)

""" # No Validation Data
train_feature = train_feature.reshape(train_feature.shape[0],48,48,1)
train_label = np_utils.to_categorical(train_label, classNum)
"""
######################### Start CNN #########################
model = Sequential()

model.add(Convolution2D(25,(3,3),input_shape=(48,48,1)))
model.add(MaxPooling2D((2,2)))
model.add(Convolution2D(50,(3,3)))
model.add(MaxPooling2D((2,2)))
model.add(Convolution2D(100,(3,3)))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())

model.add(Dropout(0.3))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(512))
model.add(Activation('relu'))

model.add(Dropout(0.3))
model.add(Dense(classNum))
model.add(Activation('softmax'))

model.summary()
model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

# Image Preprocessing - add noise
datagen = ImageDataGenerator(
    rotation_range=2.0,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip = True,
    vertical_flip = True,
    )

""" # No adding noise
#es = EarlyStopping(monitor = 'val_loss', patience = 3, verbose = 1)
#model.fit(train_feature, train_label, validation_split = 0.2, callbacks = [es], batch_size = 100, epochs = 20)
#model.fit(train_feature, train_label, validation_split = 0.1, batch_size = 100, epochs = 20)
"""
# adding noise
batchNum = 32
for i in range(4):
    model.fit(x_feature, x_label,validation_data=(valid_feature,valid_label), batch_size = batchNum, epochs = 5)
    model.fit_generator(datagen.flow(x_feature, x_label, batch_size = batchNum),   # every flow has batchNum figures
                        steps_per_epoch = x_feature.shape[0]/batchNum,
                        epochs = 5,
                        validation_data = (valid_feature, valid_label)
                        )

score = model.evaluate(train_feature, train_label)
print ("\nTrain accuracy:", score[1])

model.save(sys.argv[2])
