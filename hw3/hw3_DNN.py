################################################################################
#                             Machine Learning 2017                            #
#                      Hw3 : Image Sentiment Classification                    #
#                              Dense Neural Network                            #
#                          Description : training model                        #
#                      script : python3 train.csv model.h5                     #
################################################################################
import pandas as pd
import numpy as np
import sys
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Convolution2D, MaxPooling2D, Flatten
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
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
train_label = np_utils.to_categorical(train_label, classNum)

######################### Start DNN #########################
model = Sequential()

model.add(Dense(input_dim=48*48,units=512))
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
model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

batchNum = 100

# callbacks function
number = 1
csv_logger = CSVLogger('training' + str(number) + '.log') # store training info
save = ModelCheckpoint(sys.argv[2], monitor='val_acc', verbose=0,
                       save_best_only = True, save_weights_only=False,
                       mode='auto', period=1) # save improved model only
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=2,
                      verbose=1, mode='auto')

model.fit(train_feature, train_label, validation_split = 0.15,
          batch_size = batchNum, epochs=200, callbacks=[csv_logger, save])

model.save(sys.argv[2])

score = model.evaluate(train_feature, train_label)
print ("\nTrain accuracy:", score[1])
