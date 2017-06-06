import numpy as np
from keras.models import load_model
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
# ==== PARSING =================================================================
parser = ArgumentParser()
parser.add_argument('--old', action='store_true', help='Use old version')
parser.add_argument('-n', '--normal', action='store_true', help='use Normalization')
parser.add_argument('-m', '--model', help='model name')
parser.add_argument('-o', '--output', help='output name')
parser.add_argument('-i', '--input', help='input name')
args = parser.parse_args()

# ==== Read Test Data ==========================================================
test_path = args.input
users,movies,_ = read_data(test_path,False)
if args.old:
    users = np.array(users).astype('int')
    movies = np.array(movies).astype('int')
else:
    users = np.array(users).astype('int') - 1
    movies = np.array(movies).astype('int') - 1

model = load_model(args.model)

output = model.predict([users,movies], verbose=1)
if args.normal:
    output = output * 1.116898 + 3.581712

with open(args.output,'w') as f:
	f.write("TestDataID,Rating\n")
	for i,rating in enumerate(output):
            if round(rating[0]) > 5:
                rating[0] = 5
            elif round(rating[0] < 1):
                rating[0] = 1
            f.write("{},{}\n".format(i+1,rating[0]))
