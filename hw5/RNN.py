import numpy as np
import re
import csv
import sys
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import GRU, LSTM
from keras.layers.embeddings import Embedding
from keras.optimizers import Adamax, SGD, Adam, Adadelta
from keras import backend as K
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.models import load_model
from sklearn.preprocessing import MultiLabelBinarizer

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
    #thresh = 0.9
    #y_pred = K.cast(K.greater(y_pred,thresh),dtype='float32')
    return fbeta_score(y_true, y_pred, beta=1)

############################### main function ##################################
if __name__ == '__main__':
    record = False # record text/tag match info
    TextLength = 300
    EMBEDDING_DIM = 100
    Valid_split = 0.2
################################ data reading ##################################
    with open('train_data.csv','r') as f:
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
    with open('test_data.csv','r') as f:
        line = f.readlines()
        test_texts = [re.sub(pattern = '\d+,',
                        repl =  '',
                        string = line[i],
                        count = 1)
                 for i in range(1, len(line))]
############################# Text Preprocessing ###############################
    tokenizer = Tokenizer(split = ' ')
    tokenizer.fit_on_texts(texts + test_texts) # match word & sequence
    text_index = tokenizer.word_index # return match of word and sequence in dict type
    text_sequences = tokenizer.texts_to_sequences(texts) # convert words into sequences and return list of sequences
    test_seq = tokenizer.texts_to_sequences(test_texts)
    print('\n{} tokens in texts'.format(len(text_index)))

    x_test = pad_sequences(test_seq, maxlen = TextLength)
    traintemp = pad_sequences(text_sequences, maxlen = TextLength)
    validnum = int(Valid_split * traintemp.shape[0])
    x_val = traintemp[:validnum]
    x_train = traintemp[validnum:]

############################## Tag Preprocessing ###############################
    '''tokenizer_tags = Tokenizer(split = ' ',
                               lower = False,
                               filters = '!"#$%&()*+,./:;<=>?@[\\]^_`{|}~\t\n')
    tokenizer_tags.fit_on_texts(tags)
    tag_sequences = tokenizer_tags.texts_to_sequences(tags)
    tag_index = tokenizer_tags.word_index'''

    category = []
    label = []
    with open('label_mapping.csv','r') as f:
        for row in csv.reader(f):
            category.append(row[0])
            label.append(int(row[1]))
    tagDict = {category[i] : label[i] for i in range(38)}
    #print(tagDict)
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

############################## Record Dictionary ###############################
    if record == True:
        from operator import itemgetter
        with open('category.csv','w') as csvfile:
            tagdict = sorted(tag_index.items(), key=itemgetter(1))
            for key in tagdict:
                 csvfile.write(key[0] + ',' + str(key[1]))
                 csvfile.write('\n')
        with open('text_index.csv','w') as csvfile:
            textdict = sorted(text_index.items(), key=itemgetter(1))
            for key in textdict:
                 csvfile.write(key[0] + ',' + str(key[1]))
                 csvfile.write('\n')

############################## Embedding Layer #################################
    embeddings_index = {}
    f = open('glove.6B.%dd.txt'%EMBEDDING_DIM,'r')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('using glove with dimension(%d)' %EMBEDDING_DIM)

    embedding_matrix = np.zeros((len(text_index) + 1, EMBEDDING_DIM))
    for word, i in text_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

################################# RNN model ####################################
    model = Sequential()
    # input size : (batch_size, sequence_length)
    # output size : (batch_size, sequence_length, output_dim)
    model.add(Embedding(input_dim = len(text_index)+1, # num of tokens
                        output_dim = EMBEDDING_DIM,
                        weights=[embedding_matrix],
                        input_length = x_train.shape[1],
                        trainable=False))

    model.add(GRU(128,activation='tanh',dropout=0.1))
    model.add(Dense(256,activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(64,activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(38,activation='sigmoid'))
    model.summary()

    adam = Adam(lr=0.001,decay=1e-6,clipvalue=0.5)
    model.compile(loss='categorical_crossentropy',
                  metrics=[fmeasure],
                  optimizer=adam)

    es = EarlyStopping(monitor='val_fmeasure', patience = 20, verbose=1, mode='max')
    save = ModelCheckpoint(sys.argv[1], monitor='val_fmeasure', verbose=1,
                           save_best_only = True, save_weights_only=False,
                           mode='max', period=1)

    model.fit(x_train, y_train,
              validation_data=(x_val, y_val),
              batch_size = 128,
              epochs = 500,
              callbacks=[save,es])
              
################################## predict #####################################
    bestmodel = load_model(sys.argv[1])
    y_test = bestmodel.predict(x_test, batch_size = 128)

    thresh = 0.4
    with open(sys.argv[2],'w') as output:
        output.write('\"id\",\"tags\"\n')
        y_test_thresh = (y_test > thresh).astype('int')
        for index,labels in enumerate(y_test_thresh):
            labels = [category[i] for i,value in enumerate(labels) if value==1 ]
            labels_original = ' '.join(labels)
            output.write('\"%d\",\"%s\"\n'%(index,labels_original))
