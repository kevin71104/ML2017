import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K
import numpy as np
import sys
from sklearn.manifold import TSNE
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

def draw(embed_matrix, category):
    y = np.array(category)
    x = np.array(embed_matrix, dtype = np.float64)
    tsne = TSNE(n_components=2)
    vis_data = tsne.fit_transform(x)
    vis_x = vis_data[:,0]
    vis_y = vis_data[:,1]

    cm = plt.cm.get_cmap('RdYlBu')
    sc = plt.scatter(vis_x, vis_y, c=y, cmap = cm)
    plt.colorbar(sc)
    plt.show()


train_path = './data/train.csv'
users,movies,rating = read_data(train_path,True)
users = np.array(users).astype('int')
movies = np.array(movies).astype('int')
rating = np.array(rating).astype('float')

max_userid = np.max(users)
max_movieid = np.max(movies)
print("Find maximum user id {} and maximum movie id {} in training set..."\
	.format(max_userid,max_movieid))

r_mean = np.mean(rating)
r_std = np.std(rating)
rating = (rating - r_mean) / r_std

model = load_model(sys.argv[1])

movie_emb = np.array(model.layers[3].get_weights()).squeeze()
print('movie embedding matrix shape:', movie_emb.shape)

movies_df = pd.read_csv('data/movies.csv', sep='::', engine='python')

# get all genres of movie
all_genres = np.array([])
for genres in movies_df['Genres']:
    for genre in genres.split('|'):
        all_genres = np.append(all_genres, genre)
all_genres = np.unique(all_genres)
print('all genres:\n', all_genres)

movies_info = np.zeros((max_movieid, 6))

cat1 = ['Action', 'Adventure', 'Western']
cat2 = ['Animation', "Children's", 'Comedy', 'Romance']
cat3 = ['Crime', 'Thriller', 'Film-Noir', 'Horror', 'Mystery']
cat4 = ['Documentary', 'War']
cat5 = ['Drama', 'Musical']
cat6 = ['Fantasy', 'Sci-Fi']

# get one-hot encoding
for idx, movie_id in enumerate(movies_df['movieID']):
    genres = movies_df['Genres'][idx].split('|')
    tmp = np.zeros(6)
    for genre in genres:
        if genre in cat1:
            tmp[0] = tmp[0] + 1
        elif genre in cat2:
            tmp[1] = tmp[1] + 1
        elif genre in cat3:
            tmp[2] = tmp[2] + 1
        elif genre in cat4:
            tmp[3] = tmp[3] + 1
        elif genre in cat5:
            tmp[4] = tmp[4] + 1
        elif genre in cat6:
            tmp[5] = tmp[5] + 1
    movies_info[movie_id - 1] = tmp
movie_cat = np.argmax(movies_info, axis = 1)
print(movies_info[:20])
print(movie_cat[:20])

draw(movie_emb, movie_cat)
