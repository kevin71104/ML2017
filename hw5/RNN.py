import numpy as np
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import GRU, LSTM
from keras.layers.embeddings import Embedding
from keras.optimizers import Adamax, SGD, Adam, Adadelta
from keras import backend as K
from keras.callbacks import CSVLogger, ModelCheckpoint
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
    return fbeta_score(y_true, y_pred, beta=1)



if __name__ == '__main__':
    record = False # record text/tag match info
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

################################ Preprocessing #################################
    tokenizer = Tokenizer(split = ' ')
    # match word & sequence
    tokenizer.fit_on_texts(texts)
    # convert words into sequences and return list of sequences
    text_sequences = tokenizer.texts_to_sequences(texts)
    # return match of word and sequence in dict type
    text_index = tokenizer.word_index
    print('\n{} tokens in texts'.format(len(text_index)))

    # padding sequences to equal length by adding leading 0s
    x_train = pad_sequences(text_sequences)
    print('\nShape of data: ', x_train.shape)

    tokenizer_tags = Tokenizer(split = ' ',
                               lower = False,
                               filters = '!"#$%&()*+,./:;<=>?@[\\]^_`{|}~\t\n')
    tokenizer_tags.fit_on_texts(tags)
    tag_sequences = tokenizer_tags.texts_to_sequences(tags)
    tag_index = tokenizer_tags.word_index
    if record == True:
        from operator import itemgetter
        with open('category.csv','w') as csvfile:
            tag_index = sorted(tag_index.items(), key=itemgetter(1))
            for key in tag_index:
                 csvfile.write(key[0] + ',' + str(key[1]))
                 csvfile.write('\n')
        with open('text_index.csv','w') as csvfile:
            text_index = sorted(text_index.items(), key=itemgetter(1))
            for key in text_index:
                 csvfile.write(key[0] + ',' + str(key[1]))
                 csvfile.write('\n')

    y_train = MultiLabelBinarizer().fit_transform(tag_sequences)
    print('\nShape of label: ', y_train.shape)

################################# RNN model ####################################
    model = Sequential()
    # input size : (batch_size, sequence_length)
    # output size : (batch_size, sequence_length, output_dim)
    model.add(Embedding(input_dim = len(text_index)+1, # num of tokens
                        output_dim = 256,
                        input_length = x_train.shape[1]))
    '''
    model.add(LSTM(128,
                   dropout = 0.2,
                   recurrent_dropout = 0.2,
                   return_sequences=True))
    '''
    model.add(LSTM(128,
                   dropout = 0.2,
                   recurrent_dropout = 0.2))
    model.add(Dropout(0.25))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.25))
    model.add(Dense(256, activation = 'relu'))
    model.add(Dropout(0.25))
    model.add(Dense(38, activation = 'sigmoid'))

    model.summary()

    adamax = Adamax(lr = 0.002, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08, decay = 0.0)
    model.compile(loss = 'binary_crossentropy', metrics = [fmeasure,precision,recall], optimizer = 'SGD')

    save = ModelCheckpoint('model.h5', monitor='val_acc', verbose=0,
                           save_best_only = True, save_weights_only=False,
                           mode='auto', period=1)

    model.fit(x_train, y_train, batch_size = 128,epochs = 15, validation_split = 0.2, callbacks=[save])
