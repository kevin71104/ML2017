################################################################################
#                            Machine Learning 2017                             #
#                           Hw5 : Multi-label text                             #
#                          Recurrent Neural Network                            #
#   python3 RNN.py <-cat | -bin> <-o filename> <-m filename>                   #
################################################################################
import numpy as np
import re
import csv
import os
import pickle
import argparse
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import GRU
from keras.layers.embeddings import Embedding
from keras.layers.wrappers import Bidirectional
from keras.optimizers import Adamax, SGD, Adam, Adadelta, RMSprop
from keras import backend as K
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.models import load_model
from sklearn.preprocessing import MultiLabelBinarizer

BASE_DIR  = os.path.dirname(os.path.realpath(__file__))
GLOVE_DIR = os.path.join(BASE_DIR, 'glove')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
DATA_DIR  = os.path.join(BASE_DIR, 'data')
PRED_DIR  = os.path.join(BASE_DIR, 'prediction')
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
if not os.path.exists(PRED_DIR):
    os.makedirs(PRED_DIR)

parser = argparse.ArgumentParser()
parser.add_argument("-cat","--categorical", help="use categorical_crossentropy",
                    action="store_true")
parser.add_argument("-bin","--binary", help="use binary_crossentropy",
                    action="store_true")
parser.add_argument("-b","--bag", help="use bag of words",
                    action="store_true")
parser.add_argument("-r","--record", help="first time",
                    action="store_true")
parser.add_argument("-m","--model", help="model filename")

args = parser.parse_args()

TextLength = 300
EMBEDDING_DIM = 200
Valid_split = 0.1
TRAIN_SPLIT = 0.5
BATCHNUM = 100
DROPOUT_RATE = 0.5

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
    thresh = np.ones(38)*0.55
    thresh[0] = 0.9
    y_pred = K.cast(K.greater(y_pred,thresh),dtype='float32')
    return fbeta_score(y_true, y_pred, beta=1)

############################### main function ##################################
if __name__ == '__main__':
    ############################## data reading ################################
    print('Start reading data')
    trainfile = os.path.join(DATA_DIR, "train_data.csv")
    with open(trainfile,'r') as f:
        line = f.readlines()
        # . : any character w/o newline
        # * : match 0 or more repetitions of the preceding RE
        # ? : Adding ? after the qualifier makes as few characters as possible will be matched
        # findall will return a list
        tags = [re.findall('\"(.*?)\"', line[i])[0]
                for i in range(1, len(line))]
        texts = [re.sub(pattern = '\d+,\"(.*?)\",',
                        repl =  '',
                        string = line[i],
                        count = 1)
                 for i in range(1, len(line))]
        train = [text_to_word_sequence(row, lower=True, split=" ")
                 for row in texts]
    if args.record:
        testfile = os.path.join(DATA_DIR, "test_data.csv")
        with open(testfile,'r') as f:
            line = f.readlines()
            test_texts = [re.sub(pattern = '\d+,',
                                 repl =  '',
                                 string = line[i],
                                 count = 1)
                          for i in range(1, len(line))]
            test = [text_to_word_sequence(row, lower=True, split=" ")
                    for row in test_texts]

    ################### First Time Record TextDict TagDict #####################
    if args.record:
        tokenizer = Tokenizer(split = ' ')
        tokenizer.fit_on_texts(texts + test_texts) # match word & sequence
        textDict  = tokenizer.word_index # return match of word and sequence in dict type
        train_seq = tokenizer.texts_to_sequences(texts) # convert words into sequences and return list of sequences
        #test_seq  = tokenizer.texts_to_sequences(test_texts)
        print('Find {} tokens in texts'.format(len(textDict)))

        with open('tokenizer.pkl', 'wb') as tok_file:
            pickle.dump(tokenizer, tok_file)

        '''tokenizer_tags = Tokenizer(split = ' ',
                                   lower = False,
                                   filters = '!"#$%&()*+,./:;<=>?@[\\]^_`{|}~\t\n')
        tokenizer_tags.fit_on_texts(tags)
        tag_sequences = tokenizer_tags.texts_to_sequences(tags)
        tag_index = tokenizer_tags.word_index

        from operator import itemgetter
        with open('text_index.csv','w') as csvfile:
            textdict = sorted(text_index.items(), key=itemgetter(1))
            for key in textdict:
                 csvfile.write(key[0] + ',' + str(key[1]))
                 csvfile.write('\n')'''
    else:
        tokenizer = pickle.load(open('tokenizer.pkl','rb'))
        textDict  = tokenizer.word_index
        print('Find {} tokens in texts'.format(len(textDict)))
        train_seq = tokenizer.texts_to_sequences(texts)

    ########################### Text Preprocessing #############################
    if args.bag:
        traintemp = tokenizer.sequences_to_matrix(train_seq, mode='freq')
    else:
        traintemp = pad_sequences(train_seq, maxlen = TextLength)

    validnum = int(Valid_split * traintemp.shape[0])
    trainnum = int((Valid_split + TRAIN_SPLIT)* traintemp.shape[0])
    x_val   = traintemp[:validnum]
    x_train = traintemp[validnum:]

    ############################ Tag Preprocessing #############################
    category = []
    label = []
    with open(os.path.join(DATA_DIR, "label_mapping.csv"),'r') as f:
        for row in csv.reader(f):
            category.append(row[0])
            label.append(int(row[1]))
    tagDict = {category[i] : label[i] for i in range(38)}

    for i in range(len(tags)):
        tags[i] = tags[i].split(' ')

    tag_sequences = []
    for i in range(len(tags)):
        seq = [ tagDict[j] for j in tags[i]]
        tag_sequences.append(seq)
    tag_sequences = np.array(tag_sequences)

    tagtemp = MultiLabelBinarizer().fit_transform(tag_sequences)
    y_val = tagtemp[:validnum]
    y_train = tagtemp[validnum:]

    ############################ Embedding Layer ###############################
    if not args.bag:
        print('Start making embedding layer')
        embeddings_index = {}
        glovefile = os.path.join(GLOVE_DIR, "glove.6B.%dd.txt" %EMBEDDING_DIM)
        f = open(glovefile,'r')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        print('using glove with dimension(%d)' %EMBEDDING_DIM)

        embedding_matrix = np.zeros((len(textDict) + 1, EMBEDDING_DIM))
        for word, i in textDict.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector

    ################################# Model ####################################
    if args.categorical:
        print("use categorical_crossentropy")
        LOSS = 'categorical_crossentropy'
    elif args.binary:
        print('use binary_crossentropy')
        LOSS = 'binary_crossentropy'
    if args.bag:
        model = Sequential()
        model.add(Dense(256, activation='relu', input_shape=(x_train.shape[1], )))
        model.add(Dropout(DROPOUT_RATE))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(DROPOUT_RATE))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(DROPOUT_RATE))
        model.add(Dense(38, activation='sigmoid'))
    else:
        model = Sequential()
        # input size : (batch_size, sequence_length)
        # output size : (batch_size, sequence_length, output_dim)
        model.add(Embedding(input_dim = len(textDict)+1, # num of tokens
                            output_dim = EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length = x_train.shape[1],
                            trainable=False))

        '''model.add(Bidirectional(GRU(128, dropout = DROPOUT_RATE,
                                    recurrent_dropout=0.5,
                                    return_sequences=True)))
        model.add(Bidirectional(GRU(128, dropout = DROPOUT_RATE,
                                    recurrent_dropout=0.5)))'''
        model.add(GRU(256, dropout=0.5, recurrent_dropout=0.5, return_sequences=True))
        model.add(GRU(64, dropout=0.5, recurrent_dropout=0.5))
        model.add(Dense(256,activation='elu'))
        model.add(Dropout(0.5))
        model.add(Dense(128,activation='elu'))
        model.add(Dropout(0.5))
        model.add(Dense(64,activation='elu'))
        model.add(Dropout(0.5))
        model.add(Dense(38,activation='sigmoid'))

    model.summary()
    adam = Adam(lr=0.001,decay=1e-6,clipvalue=0.5)
    rmsprop = RMSprop(lr=0.001)
    model.compile(loss=LOSS,
                      metrics=[fmeasure],
                      optimizer=rmsprop)

    es = EarlyStopping(monitor='val_fmeasure',
                           patience = 10,
                           verbose=1,
                           mode='max')

    modelfile = os.path.join(MODEL_DIR,args.model)
    save = ModelCheckpoint(modelfile,
                           monitor='val_fmeasure',
                           verbose=1,
                           save_best_only = True,
                           save_weights_only = False,
                           mode='max',
                           period=1)

    model.fit(x_train, y_train,
              validation_data=(x_val, y_val),
              batch_size = BATCHNUM,
              epochs = 300,
              callbacks=[save,es])
