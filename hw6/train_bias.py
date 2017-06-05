import numpy as np
import sys
from keras.models import Sequential,Model, load_model
from keras.layers import Input,Embedding, Reshape, Merge, Dropout,Flatten,Dot,Add, Flatten, BatchNormalization, Concatenate, Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import regularizers
from argparse import ArgumentParser

def read_data(path,train):
    users = []
    movies = []
    rating = []
    with open(path,'r') as f:
        for line in f.readlines():
            line = line.strip().split(',')
            if line[1] == 'UserID':
                continue
            else:
                users.append(line[1])
                movies.append(line[2])
                if train:
                    rating.append(line[3])
    return users,movies,rating

def split_data(U,M,R,split_ratio):
    np.random.seed(0)
    indices = np.random.permutation(users.shape[0])

    U_data = U[indices]
    M_data = M[indices]
    R_data = R[indices]

    num_validation_sample = int(split_ratio * U_data.shape[0] )

    U_train = U_data[num_validation_sample:]
    M_train = M_data[num_validation_sample:]
    R_train = R_data[num_validation_sample:]

    U_val = U_data[:num_validation_sample]
    M_val = M_data[:num_validation_sample]
    R_val = R_data[:num_validation_sample]

    return (U_train,M_train,R_train),(U_val,M_val,R_val)

# ==== parsing arguments =======================================================
parser = ArgumentParser()
parser.add_argument('--dnn', action='store_true', help='Use DNN model')
parser.add_argument('-n', '--normal', action='store_true', help='use Normalization')
parser.add_argument('-m', '--model', help='model name')
parser.add_argument('-o', '--output', help='output name')
args = parser.parse_args()

# ==== Read Dataset ============================================================
train_path = './data/train.csv'
users,movies,rating = read_data(train_path,True)
users = np.array(users).astype('int')
movies = np.array(movies).astype('int')
rating = np.array(rating).astype('float')

if args.normal:
    r_mean = np.mean(rating)
    r_std = np.std(rating)
    rating = (rating - r_mean) / r_std
    print('rating mean(%f) and rating std(%f)'%(r_mean,r_std))

max_userid = np.max(users)
max_movieid = np.max(movies)
print("Find maximum user id {} and maximum movie id {} in training set..."\
	.format(max_userid,max_movieid))

embedding_dim = 10
split_ratio = 0.1
DROPOUT_RATE = 0.1

User_input = Input(shape = (1,),dtype = 'int32')
User_embed = Embedding(output_dim = embedding_dim,
                       input_dim = max_userid,
                       input_length = 1,
                       embeddings_initializer='random_normal',
                       embeddings_regularizer = regularizers.l2(1e-5),
                       trainable=True)(User_input)
User_reshape = Reshape((embedding_dim,))(User_embed)
User_reshape = Dropout(DROPOUT_RATE)(User_reshape)
User_bias = (Embedding(output_dim = 1,
                       input_dim = max_userid,
                       input_length = 1,
                       embeddings_initializer='zeros',
                       embeddings_regularizer = regularizers.l2(1e-5),
                       trainable=True)(User_input))
User_bias = Flatten()(User_bias)

Movie_input = Input(shape = (1,),dtype = 'int32')
Movie_embed = Embedding(output_dim = embedding_dim,
                        input_dim = max_movieid,
                        input_length = 1,
                        embeddings_initializer='random_normal',
                        embeddings_regularizer = regularizers.l2(1e-5),
                        trainable=True)(Movie_input)
Movie_reshape = Reshape((embedding_dim,))(Movie_embed)
Movie_reshape = Dropout(DROPOUT_RATE)(Movie_reshape)
Movie_bias = (Embedding(output_dim = 1,
                        input_dim = max_userid,
                        input_length = 1,
                        embeddings_initializer='zeros',
                        embeddings_regularizer = regularizers.l2(1e-5),
                        trainable=True)(Movie_input))
Movie_bias = Flatten()(Movie_bias)

if args.dnn:
    concat = Concatenate()([User_reshape, Movie_reshape])
    dnn = Dense(256, activation='relu')(concat)
    dnn = Dropout(DROPOUT_RATE)(dnn)
    dnn = BatchNormalization()(dnn)
    dnn = Dense(256, activation='relu')(dnn)
    dnn = Dropout(DROPOUT_RATE)(dnn)
    dnn = BatchNormalization()(dnn)
    dnn = Dense(256, activation='relu')(dnn)
    dnn = Dropout(DROPOUT_RATE)(dnn)
    dnn = BatchNormalization()(dnn)
    output = Dense(1, activation='relu')(dnn)
    model = Model(inputs=[User_input, Movie_input], outputs = output)
else:
    Main_dot = Dot(axes = 1)([User_reshape, Movie_reshape])
    Main_add = Add()([Main_dot, User_bias, Movie_bias])
    model = Model([User_input,Movie_input], Main_add)

model.summary()
checkpoint = ModelCheckpoint(filepath = 'bestmodel%s.hdf5'%args.model,
                             verbose = 1,
                             save_best_only = True,
                             save_weights_only = False,
                             monitor = 'val_loss',
                             mode = 'auto')

es = EarlyStopping(monitor='val_loss',
                   patience = 5,
                   verbose=1,
                   mode='min')

model.compile(loss = 'mse',optimizer = 'adam')

(U_train, M_train, R_train),(U_val, M_val, R_val) = split_data(users,movies,rating,split_ratio)
model.fit([U_train,M_train],R_train,
          batch_size = 128,
          epochs = 80,
          validation_data = ([U_val,M_val],R_val),
          callbacks = [checkpoint,es])

model = load_model('bestmodel%s.hdf5'%args.model)

test_path = './data/test.csv'
users,movies,_ = read_data(test_path,False)
users = np.array(users).astype('int')
movies = np.array(movies).astype('int')

output = model.predict([users,movies], verbose=1)
if args.normal:
    output = output * r_std + r_mean

with open(args.output,'w') as f:
	f.write("TestDataID,Rating\n")
	for i,rating in enumerate(output):
            if round(rating[0]) > 5:
                rating[0] = 5
            elif round(rating[0] < 1):
                rating[0] = 1
            f.write("{},{}\n".format(i+1,rating[0]))
