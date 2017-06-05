import numpy as np
import sys
from keras.models import Sequential,Model, load_model
from keras.layers import Input,Embedding, Reshape, Merge, Dropout,Flatten,Dot,Add, Flatten, BatchNormalization, Concatenate, Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import regularizers
from argparse import ArgumentParser
import pandas as pd

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
parser.add_argument('-e', '--extra', action='store_true', help='use extra feature')
parser.add_argument('-m', '--model', help='model name')
parser.add_argument('-o', '--output', help='output name')
parser.add_argument('-d', '--dimension', help='latent dimension')
parser.add_argument('-b', '--bias', action='store_true', help='use bias')
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

if args.extra:
    users_df = pd.read_csv('data/users.csv', sep='::', engine='python')
    users_age = (users_df['Age'] - np.mean(users_df['Age'])) / np.std(users_df['Age'])

    movies_df = pd.read_csv('data/movies.csv', sep='::', engine='python')

    # get all genres of movie
    all_genres = np.array([])
    for genres in movies_df['Genres']:
        for genre in genres.split('|'):
            all_genres = np.append(all_genres, genre)
    all_genres = np.unique(all_genres)
    #print('all genres:\n', all_genres)

    movies_info = np.zeros((max_movieid, all_genres.shape[0]))
    users_info = np.zeros((max_userid, 23))

    for idx, user_id in enumerate(users_df['UserID']):
        gender = 1 if users_df['Gender'][idx] == 'M' else 0
        occu = np.zeros(np.max(np.unique(users_df['Occupation'])) + 1)
        occu[users_df['Occupation'][idx]] = 1
        tmp = [gender, users_age[idx]]
        tmp.extend(occu)
        users_info[user_id - 1] = tmp

    for idx, movie_id in enumerate(movies_df['movieID']):
        genres = movies_df['Genres'][idx].split('|')
        tmp = np.zeros(all_genres.shape[0])
        for genre in genres:
            tmp[np.where(all_genres == genre)[0][0]] = 1
        movies_info[movie_id - 1] = tmp

# ==== Model Parameter =========================================================
embedding_dim = int(args.dimension)
split_ratio = 0.1
DROPOUT_RATE = 0.3

# ==== Model ===================================================================
User_input = Input(shape = (1,),dtype = 'int32')
User_embed = Embedding(output_dim = embedding_dim,
                       input_dim = max_userid,
                       input_length = 1,
                       embeddings_initializer='random_normal',
                       embeddings_regularizer = regularizers.l2(1e-5),
                       trainable=True)(User_input)
User_reshape = Reshape((embedding_dim,))(User_embed)
User_reshape = Dropout(0.1)(User_reshape)
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
Movie_reshape = Dropout(0.1)(Movie_reshape)
Movie_bias = (Embedding(output_dim = 1,
                        input_dim = max_userid,
                        input_length = 1,
                        embeddings_initializer='zeros',
                        embeddings_regularizer = regularizers.l2(1e-5),
                        trainable=True)(Movie_input))
Movie_bias = Flatten()(Movie_bias)

if args.dnn:
    print('Use dnn model without extra feature')
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
elif args.extra:
    print('Use dnn model with extra feature')
    U_info_emb = Embedding(input_dim = max_userid,
                           output_dim = users_info.shape[1],
                           weights=[users_info],
                           trainable=False)(User_input)
    U_info_emb = Flatten()(U_info_emb)

    M_info_emb = Embedding(input_dim = max_movieid,
                           output_dim = movies_info.shape[1],
                           weights=[movies_info],
                           trainable=False)(Movie_input)
    M_info_emb = Flatten()(M_info_emb)

    concat = Concatenate()([User_reshape, Movie_reshape, U_info_emb, M_info_emb])
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
    print('Use matrix factorization')
    Main_dot = Dot(axes = 1)([User_reshape, Movie_reshape])
    if args.bias:
        Main_add = Add()([Main_dot, User_bias, Movie_bias])
    else:
        Main_add = Main_dot
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

# ==== Validation Set and Train Set ============================================
(U_train, M_train, R_train),(U_val, M_val, R_val) = split_data(users,movies,rating,split_ratio)
model.fit([U_train,M_train],R_train,
          batch_size =128,
          epochs = 80,
          validation_data = ([U_val,M_val],R_val),
          callbacks = [checkpoint,es])

model = load_model('bestmodel%s.hdf5'%args.model)

# ==== Test Set ================================================================
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
