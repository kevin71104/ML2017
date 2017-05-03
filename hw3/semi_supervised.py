################################################################################
#                            Machine Learning 2017                             #
#                     Hw3 : Image Sentiment Classification                     #
#                         Convolutional Neural Network                         #
#                 Description : semi-supervised training model                 #
#  script : python3 semi_supervised.py train.csv model.h5 <0 or 1> [test.csv]  #
################################################################################
#import pandas as pd
import csv
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
train_label = []
train_feature = []
with open(sys.argv[1],'r') as train_file:
    for row in csv.DictReader(train_file):
        train_feature.append(row['feature'].split())
        train_label.append(row['label'])
train_feature = np.array(train_feature).astype('float32')
train_feature /= 255
train_label = np.array(train_label).astype('int')

classNum = 7
validNum = 5000
unlabelNum = 5000
randvalid = 0
semi = int(sys.argv[3])
test = (len(sys.argv) == 5)
if len(sys.argv) == 5:
    testfile = sys.argv[4]

############################### validation Data ################################
if semi == 1:
    bound = 0.9
    if test == 1:
        unlabel_feature = []
        with open(testfile,'r') as train_file:
            for row in csv.DictReader(train_file):
                unlabel_feature.append(row['feature'].split())
        unlabel_feature = np.array(unlabel_feature).astype('float32')
        unlabel_feature = unlabel_feature/255
        valid_label = train_label[:validNum]
        valid_feature = train_feature[:validNum]
        x_feature = train_feature[validNum:]
        x_label = train_label[validNum:]
    else:
        if randvalid == 1:
            choose = rand.sample(range(0,train_feature.shape[0]-1),
                                 (validNum+unlabelNum))
            unlabel_label = train_label[choose]
            unlabel_feature = train_feature[choose]
            valid_label = unlabel_label[:validNum]
            valid_feature = unlabel_feature[:validNum]
            unlabel_feature = unlabel_feature[validNum:]
            unlabel_label = []
            x_label = np.delete(train_label,choose,axis = 0)
            x_feature = np.delete(train_feature,choose,axis = 0)
        else:
            valid_label = train_label[:validNum]
            valid_feature = train_feature[:validNum]
            unlabel_feature = train_feature[validNum : (validNum+unlabelNum)]
            x_feature = train_feature[(validNum+unlabelNum):]
            x_label = train_label[(validNum+unlabelNum):]
else:
    if randvalid == 1:
        choose = rand.sample(range(0,train_feature.shape[0]-1),validNum)
        valid_label = train_label[choose]
        valid_feature = train_feature[choose]
        x_label = np.delete(train_label,choose,axis = 0)
        x_feature = np.delete(train_feature,choose,axis = 0)
    else:
        valid_label = train_label[:validNum]
        valid_feature = train_feature[:validNum]
        x_feature = train_feature[(validNum+unlabelNum):]
        x_label = train_label[(validNum+unlabelNum):]
train_label = []
train_feature = []

############################## change input shape ##############################
x_feature = x_feature.reshape(x_feature.shape[0],48,48,1)
valid_feature = valid_feature.reshape(valid_feature.shape[0],48,48,1)
x_label = np_utils.to_categorical(x_label, classNum)
valid_label = np_utils.to_categorical(valid_label, classNum)
if semi == 1:
    unlabel_feature = unlabel_feature.reshape(unlabel_feature.shape[0],48,48,1)

################################## CNN Model ###################################
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
model.compile(loss='categorical_crossentropy', optimizer="adamax",
              metrics=['accuracy'])

#model = load_model('model31.h5')

################################## Callbacks ###################################
# save improved model only
save = ModelCheckpoint(sys.argv[2], monitor='val_acc', verbose=0,
                       save_best_only = True, save_weights_only=False,
                       mode='auto', period=1)

# store training info
csv_logger = CSVLogger('{}.log'.format(sys.argv[2][:-3]), append = True)

################################## Training ####################################
batchNum = 100
# training with data only
model.fit(x_feature, x_label,validation_data=(valid_feature,valid_label),
              batch_size = batchNum, epochs = 4, callbacks=[save, csv_logger])

# Image Preprocessing - add noise
datagen = ImageDataGenerator(
    rotation_range=10.0,
    width_shift_range=0.1,
    height_shift_range=0.1)

# training with data and noise
for i in range(0):
    # every flow has batchNum figures
    model.fit_generator(datagen.flow(x_feature, x_label, batch_size = batchNum),
                        steps_per_epoch = x_feature.shape[0]/batchNum,
                        epochs = 4,
                        validation_data = (valid_feature, valid_label),
                        callbacks=[save, csv_logger])
    model.fit(x_feature, x_label,validation_data=(valid_feature,valid_label),
              batch_size = batchNum, epochs = 2, callbacks=[save, csv_logger])

################################ Semi-Supervised ###############################
for i in range(2):
    if semi == 1:
        if (unlabel_feature.shape[0]):
            prob = model.predict(unlabel_feature)
            label = prob.argmax(1)
            decision = prob.max(1) > bound
            label = label[decision]
            feature = unlabel_feature[decision,:,:,:]
            unlabel_feature = unlabel_feature[~decision,:,:,:]
            print('add size : ' + str(feature.shape[0]))
            print('left size : '+ str(unlabel_feature.shape[0]))
            x_feature = np.concatenate((x_feature,feature),axis=0)
            label = np_utils.to_categorical(label, classNum)
            x_label = np.concatenate((x_label, label), axis=0)

    model.fit_generator(datagen.flow(x_feature, x_label, batch_size = batchNum),
                        steps_per_epoch = x_feature.shape[0]/batchNum,
                        epochs = 2,
                        validation_data = (valid_feature, valid_label),
                        callbacks=[save, csv_logger])
    model.fit(x_feature, x_label,validation_data=(valid_feature,valid_label),
              batch_size = batchNum, epochs = 1, callbacks=[save, csv_logger])

######################## Record train/valid accuracy ###########################
score = model.evaluate(x_feature, x_label)
print ("\nTrain accuracy:", score[1])
score2 = model.evaluate(valid_feature, valid_label)
print ("\nValid accuracy:", score2[1])
