################################################################################
#                            Machine Learning 2017                             #
#                     Hw3 : Image Sentiment Classification                     #
#                         Convolutional Neural Network                         #
#                         Description : training model                         #
#                 script : python3 hw3_CNN.py train.csv model.h5               #
################################################################################
import pandas as pd
import numpy as np
import random as rand
import sys
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Convolution2D, MaxPooling2D, Flatten
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

################################## Read File ###################################
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
validNum = 5000
randvalid = 0
############################### validation Data ################################
if randvalid == 1:
    choose = rand.sample(range(0,train_feature.shape[0]-1),validNum)
    valid_label = train_label[choose]
    valid_feature = train_feature[choose]
    x_label = np.delete(train_label,choose,axis = 0)
    x_feature = np.delete(train_feature,choose,axis = 0)
else:
    valid_label = train_label[:validNum]
    valid_feature = train_feature[:validNum]
    x_feature = train_feature[validNum:]
    x_label = train_label[validNum:]
train_label = []
train_feature = []

############################## change input shape ##############################
x_feature = x_feature.reshape(x_feature.shape[0],48,48,1)
valid_feature = valid_feature.reshape(valid_feature.shape[0],48,48,1)
x_label = np_utils.to_categorical(x_label, classNum)
valid_label = np_utils.to_categorical(valid_label, classNum)

################################## Start CNN ###################################
model = Sequential()

model.add(Convolution2D(32,(3,3), input_shape=(48,48,1)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add( MaxPooling2D(pool_size=(2, 2)) )
model.add(Convolution2D(64,(3,3), activation='relu'))
model.add(BatchNormalization())
model.add( MaxPooling2D(pool_size=(2, 2)) )
model.add(Convolution2D(128,(3,3), activation='relu'))
model.add(BatchNormalization())
model.add( MaxPooling2D(pool_size=(2, 2)) )

model.add(Flatten())

model.add(Dropout(0.4))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Dropout(0.4))
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Dropout(0.3))
model.add(Dense(classNum))
model.add(Activation('softmax'))

model.summary()
model.compile(loss='categorical_crossentropy', optimizer="adamax", metrics=['accuracy'])

# Image Preprocessing - add noise
datagen = ImageDataGenerator(
    rotation_range=10.0,
    width_shift_range=0.1,
    height_shift_range=0.1
    )

batchNum = 100
number = 1
csv_logger = CSVLogger('training' + str(number) + '.log') # store training info
save = ModelCheckpoint(sys.argv[2], monitor='val_acc', verbose=0,
                       save_best_only = True, save_weights_only=False,
                       mode='auto', period=1) # save improved model only
model.fit(x_feature, x_label,validation_data=(valid_feature,valid_label),
              batch_size = batchNum, epochs = 20, callbacks=[csv_logger, save])

for i in range(30):
    number = number + 1
    csv_logger = CSVLogger('training' + str(number) + '.log')
    # every flow has batchNum figures
    model.fit_generator(datagen.flow(x_feature, x_label, batch_size = batchNum),
                        steps_per_epoch = x_feature.shape[0]/batchNum,
                        epochs = 4,
                        validation_data = (valid_feature, valid_label),
                        callbacks=[csv_logger, save]
                        )
    number = number + 1
    csv_logger = CSVLogger('training' + str(number) + '.log') # store training info
    model.fit(x_feature, x_label,validation_data=(valid_feature,valid_label),
              batch_size = batchNum, epochs = 2, callbacks=[csv_logger, save])

score = model.evaluate(x_feature, x_label)
print ("\nTrain accuracy:", score[1])
score2 = model.evaluate(valid_feature, valid_label)
print ("\nValid accuracy:", score2[1])
