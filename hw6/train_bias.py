import numpy as np
import sys
from keras.models import Sequential,Model
from keras.layers import Input,Embedding, Reshape, Merge, Dropout,Flatten,Dot,Add
from keras.callbacks import ModelCheckpoint

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

train_path = './data/train.csv'
users,movies,rating = read_data(train_path,True)
users = np.array(users).astype('int')
movies = np.array(movies).astype('int')
rating = np.array(rating).astype('int')

max_userid = np.max(users)
max_movieid = np.max(movies)

print("Find maximum user id {} and maximum movie id {} in training set..."\
	.format(max_userid,max_movieid))

embedding_dim = 10
split_ratio = 0.1

def split_data(U,M,R,split_ratio):
    indices = np.arange(U.shape[0])
    np.random.shuffle(indices)

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


User_input = Input(shape = (1,),dtype = 'int32')
User_embed = Embedding(output_dim = embedding_dim,input_dim = max_userid,input_length = 1)(User_input)
User_reshape = Reshape((embedding_dim,))(User_embed)
User_bias = Reshape((1,))(Embedding(output_dim = 1,input_dim = max_userid,input_length = 1)(User_input))


Movie_input = Input(shape = (1,),dtype = 'int32')
Movie_embed = Embedding(output_dim = embedding_dim,input_dim = max_movieid,input_length = 1)(Movie_input)
Movie_reshape = Reshape((embedding_dim,))(Movie_embed)
Movie_bias = Reshape((1,))(Embedding(output_dim = 1,input_dim = max_userid,input_length = 1)(Movie_input))

Main_dot = Dot(axes = 1)([User_reshape,Movie_reshape])
Main_add = Add()([Main_dot,User_bias])
Main_add = Add()([Main_add,Movie_bias])

model = Model([User_input,Movie_input],Main_add)
model.summary()

checkpoint = ModelCheckpoint(filepath = 'best_bias.hdf5',verbose = 1,
        save_best_only = True, save_weights_only = False,monitor = 'val_loss',mode = 'auto')

model.compile(loss = 'mse',optimizer = 'adam')

(U_train,M_train,R_train),(U_val,M_val,R_val) = split_data(users,movies,rating,split_ratio)
model.fit([U_train,M_train],R_train,epochs = 30,validation_data = ([U_val,M_val],R_val),callbacks = [checkpoint])

model.save('adam_bias.h5')

test_path = './data/test.csv'
users,movies,_ = read_data(test_path,False)
users = np.array(users).astype('int')
movies = np.array(movies).astype('int')
output = model.predict([users,movies])
with open('ans_adam_bias.csv','w') as f:
	f.write("TestDataID,Rating\n")
	for i,rating in enumerate(output):
            if round(rating[0]) > 5:
                rating[0] = 5
            f.write("{},{}\n".format(i+1,rating[0]))
