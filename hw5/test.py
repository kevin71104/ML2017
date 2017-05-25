################################################################################
#                            Machine Learning 2017                             #
#                           Hw5 : Multi-label text                             #
#                          Recurrent Neural Network                            #
#   python3 test.py <-i filename> <-o filename>  [-v]                          #
################################################################################
import numpy as np
import re
import csv
import os
import pickle
import argparse
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model, save_model
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers.recurrent import GRU
from keras.layers.embeddings import Embedding
from keras.layers.wrappers import Bidirectional
from keras import backend as K
from keras.optimizers import RMSprop

def precision(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def fbeta_score(y_true, y_pred, beta=1):
    """Computes the F score.
    The F score is the weighted harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.
    This is useful for multi-label classification, where input samples can be
    classified as sets of labels. By only using accuracy (precision) a model
    would achieve a perfect score by simply assigning every class to every
    input. In order to avoid this, a metric should penalize incorrect class
    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
    computes this, as a weighted mean of the proportion of correct class
    assignments vs. the proportion of incorrect class assignments.
    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
    correct classes becomes more important, and with beta > 1 the metric is
    instead weighted towards penalizing incorrect class assignments.
    """
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score

def fmeasure(y_true, y_pred):
    """Computes the f-measure, the harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.
    """
    thresh = np.ones(38)*0.5
    thresh[0] = 0.6
    y_pred = K.cast(K.greater(y_pred,thresh),dtype='float32')
    return fbeta_score(y_true, y_pred, beta=1)

############################ file path & arg parse #############################
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')

parser = argparse.ArgumentParser()
parser.add_argument("-b","--bag", help="use bag of words",
                    action="store_true")
parser.add_argument("-v","--vote", help="use voting",
                    action="store_true")
parser.add_argument("-i","--input",  help="input filename")
parser.add_argument("-o","--output", help="output filename")
#parser.add_argument("-m","--model",  help="model filename")
args = parser.parse_args()

############################## Parameter Setting ###############################
TextLength = 300
EMBEDDING_DIM = 200
BATCHNUM = 100
THRESHOLD = 0.55
Valid_split = 0.1
DROPOUT_RATE = 0.5

################################## Test Data ###################################
#testfile = os.path.join(BASE_DIR, "test_data.csv")
testfile = args.input
with open(testfile,'r') as f:
    line = f.readlines()
    test_texts = [re.sub(pattern = '\d+,',
                         repl =  '',
                         string = line[i],
                         count = 1)
                  for i in range(1, len(line))]

############################### Text & Tag Dict ################################
tokenizer = pickle.load(open('tokenizer.pkl','rb'))
test_seq  = tokenizer.texts_to_sequences(test_texts)
textDict = tokenizer.word_index

if args.bag:
    pre_x_test = tokenizer.sequences_to_matrix(test_seq, mode='freq')
else:
    pre_x_test = pad_sequences(test_seq, maxlen = TextLength)

category = []
label = []
with open(os.path.join(BASE_DIR,'label_mapping.csv'),'r') as f:
    for row in csv.reader(f):
        category.append(row[0])
        label.append(int(row[1]))
tagDict = {category[i] : label[i] for i in range(38)}

############################# Load Model Weights ###############################
modelbag = Sequential()
modelbag.add(Dense(256, activation='relu', input_shape=(pre_x_test.shape[1], )))
modelbag.add(Dropout(DROPOUT_RATE))
modelbag.add(Dense(128, activation='relu'))
modelbag.add(Dropout(DROPOUT_RATE))
modelbag.add(Dense(64, activation='relu'))
modelbag.add(Dropout(DROPOUT_RATE))
modelbag.add(Dense(38, activation='sigmoid'))

model1 = Sequential()
model1.add(Embedding(input_dim = len(textDict)+1, # num of tokens
                    output_dim = EMBEDDING_DIM,
                    input_length = pre_x_test.shape[1],
                    trainable=False))

model1.add(Bidirectional(GRU(128, dropout = DROPOUT_RATE,
                            recurrent_dropout=0.5,
                            return_sequences=True)))
model1.add(Bidirectional(GRU(128, dropout = DROPOUT_RATE,
                            recurrent_dropout=0.5)))
model1.add(Dense(256,activation='elu'))
model1.add(Dropout(0.5))
model1.add(Dense(128,activation='elu'))
model1.add(Dropout(0.5))
model1.add(Dense(128,activation='elu'))
model1.add(Dropout(0.5))
model1.add(Dense(38,activation='sigmoid'))

model2 = Sequential()
model2.add(Embedding(input_dim = len(textDict)+1, # num of tokens
                    output_dim = EMBEDDING_DIM,
                    input_length = pre_x_test.shape[1],
                    trainable=False))

model2.add(Bidirectional(GRU(128, dropout = DROPOUT_RATE,
                            recurrent_dropout=0.5,
                            return_sequences=True)))
model2.add(GRU(256, dropout = DROPOUT_RATE,
                            recurrent_dropout=0.5))
model2.add(Dense(256,activation='elu'))
model2.add(Dropout(0.5))
model2.add(Dense(128,activation='elu'))
model2.add(Dropout(0.5))
model2.add(Dense(128,activation='elu'))
model2.add(Dropout(0.5))
model2.add(Dense(38,activation='sigmoid'))

model3 = Sequential()
model3.add(Embedding(input_dim = len(textDict)+1, # num of tokens
                    output_dim = EMBEDDING_DIM,
                    input_length = pre_x_test.shape[1],
                    trainable=False))

model3.add(GRU(256, dropout = DROPOUT_RATE,
                            recurrent_dropout=0.5,
                            return_sequences=True))
model3.add(GRU(256, dropout = DROPOUT_RATE,
                            recurrent_dropout=0.5))
model3.add(Dense(256,activation='elu'))
model3.add(Dropout(0.5))
model3.add(Dense(128,activation='elu'))
model3.add(Dropout(0.5))
model3.add(Dense(128,activation='elu'))
model3.add(Dropout(0.5))
model3.add(Dense(38,activation='sigmoid'))

if args.vote:
    thresh = np.ones(38)*THRESHOLD
    thresh[0] = 0.9
    #submodel = [ '2biGRU.h5','1biGRU.h5','cat14.h5','cat15.h5','cat16.h5']
    submodel = [ 'cat9.h5','cat14.h5','cat15.h5','cat16.h5','cat17.h5','cat18.h5','cat19.h5']
    pre_tests = []
    for i, index in enumerate(submodel):
        modelfile = './model/%s'%index
        print('\n%d: %s'%(i,index))
        if index == '2biGRU.h5':
            model1.load_weights(modelfile)
            pre_y_test = model1.predict(pre_x_test, batch_size = BATCHNUM, verbose = 1)
        elif index == '1biGRU.h5':
            model2.load_weights(modelfile)
            pre_y_test = model2.predict(pre_x_test, batch_size = BATCHNUM, verbose = 1)
        else:
            model = load_model(modelfile, custom_objects = {'fmeasure': fmeasure})
            pre_y_test = model.predict(pre_x_test, batch_size = BATCHNUM, verbose = 1)

        y_test_thresh = (pre_y_test > thresh).astype('int')
        pre_tests.append(y_test_thresh)
    pre_tests = np.array(pre_tests)
    print(pre_tests.shape)

    y_test = np.sum(pre_tests, axis=0)
    print(y_test.shape)
    '''for row in y_test:
        print(row[0], end = ', ')'''
    vote = len(submodel)/2
    print('\nvote threshold: %f'%vote)
    y_final = (y_test >= vote).astype('int')
else:
    thresh = np.ones(38)*THRESHOLD
    thresh[0] = 0.9
    model = load_model('./model/cat16.h5', custom_objects = {'fmeasure': fmeasure})
    print('Successfully loading model')
    pre_y_test = model.predict(pre_x_test, batch_size = BATCHNUM, verbose = 1)
    y_final = (pre_y_test > thresh).astype('int')

#outputfile = os.path.join(BASE_DIR,args.output)
outputfile = args.output
with open(outputfile,'w') as output:
    output.write('\"id\",\"tags\"\n')
    for index,labels in enumerate(y_final):
        labels = [category[i] for i,value in enumerate(labels) if value==1 ]
        labels_original = ' '.join(labels)
        output.write('\"%d\",\"%s\"\n'%(index,labels_original))
