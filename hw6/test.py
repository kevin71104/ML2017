import numpy as np
import sys
from keras.models import load_model

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

test_path = './data/test.csv'
users,movies,_ = read_data(test_path,False)
users = np.array(users).astype('int')
movies = np.array(movies).astype('int')

model = load_model(sys.argv[1])

output = model.predict([users,movies])
output = output * 1.116898 + 3.581712

with open(sys.argv[2],'w') as f:
	f.write("TestDataID,Rating\n")
	for i,rating in enumerate(output):
            if round(rating[0]) > 5:
                rating[0] = 5
            elif round(rating[0] < 1):
                rating[0] = 1
            f.write("{},{}\n".format(i+1,rating[0]))
