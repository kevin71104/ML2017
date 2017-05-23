################################################################################
#                            Machine Learning 2017                             #
#                           Hw5 : Multi-label text                             #
#                          Recurrent Neural Network                            #
#   python3 test.py <-o filename> <-m filename> [-v]                           #
################################################################################

import numpy as np
import re
import csv
import os
import argparse
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers.recurrent import GRU, LSTM
from keras.layers.embeddings import Embedding
from keras.optimizers import Adamax, SGD, Adam, Adadelta

############################ file path & arg parse #############################
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')

parser = argparse.ArgumentParser()
parser.add_argument("-v","--validation", help="check validation set",
                    action="store_true")
parser.add_argument("-o","--output", help="output filename")
parser.add_argument("-m","--model", help="model filename")
args = parser.parse_args()

############################## Parameter Setting ###############################
modelfile = os.path.join(MODEL_DIR,args.model)
TextLength = 300
EMBEDDING_DIM = 200
BATCHNUM = 100
THRESHOLD = 0.55
Valid_split = 0.1

################################## Test Data ###################################
if args.validation:
    testfile = os.path.join(BASE_DIR, "train_data.csv")
    with open(testfile,'r') as f:
        line = f.readlines()
        test_texts = [re.sub(pattern = '\d+,\"(.*?)\",',
                             repl =  '',
                             string = line[i],
                             count = 1)
                      for i in range(1, len(line))]
    '''validnum = int(Valid_split * len(test_texts))
    test = [text_to_word_sequence(row, lower=True, split=" ")
            for row in test_texts[:validnum]]'''
    test = [text_to_word_sequence(row, lower=True, split=" ")
            for row in test_texts]

else:
    testfile = os.path.join(BASE_DIR, "test_data.csv")
    with open(testfile,'r') as f:
        line = f.readlines()
        test_texts = [re.sub(pattern = '\d+,',
                             repl =  '',
                             string = line[i],
                             count = 1)
                      for i in range(1, len(line))]
    test = [text_to_word_sequence(row, lower=True, split=" ")
            for row in test_texts]

############################### Text & Tag Dict ################################
text = []
label = []
with open(os.path.join(BASE_DIR, "text_index.csv"),'r') as f:
    for row in csv.DictReader(f):
        text.append(row['text'])
        label.append(int(row['label']))
textDict = {text[i] : label[i] for i in range(len(text))}

category = []
label = []
with open(os.path.join(BASE_DIR,'label_mapping.csv'),'r') as f:
    for row in csv.reader(f):
        category.append(row[0])
        label.append(int(row[1]))
tagDict = {category[i] : label[i] for i in range(38)}

############################# Word2Seq & pad_seq ###############################
test_seq = [[textDict[entry] for entry in row ] for row in test]
x_test = pad_sequences(test_seq, maxlen = TextLength)

############################# Load Model Weights ###############################
model = Sequential()
# input size : (batch_size, sequence_length)
# output size : (batch_size, sequence_length, output_dim)
model.add(Embedding(input_dim = len(textDict)+1, # num of tokens
                    output_dim = EMBEDDING_DIM,
                    input_length = TextLength,
                    trainable=False))
model.add(GRU(256, dropout=0.5, recurrent_dropout=0.5, return_sequences=True))
model.add(GRU(256, dropout=0.5, recurrent_dropout=0.5))
model.add(Dense(256,activation='elu'))
model.add(Dropout(0.5))
model.add(Dense(128,activation='elu'))
model.add(Dropout(0.5))
model.add(Dense(64,activation='elu'))
model.add(Dropout(0.5))
model.add(Dense(38,activation='sigmoid'))
model.summary()
model.load_weights(modelfile)

############################# Predict and Record ###############################
y_test = model.predict(x_test, batch_size = BATCHNUM, verbose = 1)
for i in range(10):
    print(y_test[i])
thresh = THRESHOLD
with open(args.output,'w') as output:
    output.write('\"id\",\"tags\"\n')
    y_test_thresh = (y_test > thresh).astype('int')
    for index,labels in enumerate(y_test_thresh):
        labels = [category[i] for i,value in enumerate(labels) if value==1 ]
        labels_original = ' '.join(labels)
        output.write('\"%d\",\"%s\"\n'%(index,labels_original))
