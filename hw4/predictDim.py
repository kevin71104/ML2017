import numpy as np
import csv
from sklearn.metrics.pairwise import euclidean_distances

data = np.load('data.npz')

with open('stat.csv','r')as csvfile:
    for row in csv.reader(csvfile):
        w = (list(map(float,row)))
print(w)

pred=[]
for i in range(200):
    print(i)
    x = data[str(i)]
    #print(x.shape)
    dist = euclidean_distances(x[:10000])
    dist.partition(1, axis=1)
    dist_min = dist[:,1]
    std = np.std(dist_min)
    dim = round(np.exp((std - w[1]) / w[0]))
    if dim > 60:
        dim = 60
    pred.append(dim)
pred = np.log(pred)
#print(pred)

with open('pred.csv','w')as csvfile:
    csvfile.write('SetId,LogDim\n')
    for i in range(pred.shape[0]):
        csvfile.write(str(i)+ ',' + str(pred[i]) + '\n')
