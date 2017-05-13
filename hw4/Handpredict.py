import numpy as np
import csv
from PIL import Image as im
from sklearn.metrics.pairwise import euclidean_distances


filepath = './hand/hand.seq'

data = []
for i in range(1,482):
        filename = filepath + str(i) + '.png'
        image = im.open(filename)
        image = image.resize(size=(10,10),resample=im.BILINEAR)
        #image.show()
        arr = np.asarray(image)
        data.append(arr)

data = np.array(data,'float32')
data=data.reshape(data.shape[0],data.shape[1]*data.shape[2])
print(data.shape)

with open('stat.csv','r')as csvfile:
    for row in csv.reader(csvfile):
        w = (list(map(float,row)))

#pred=[]

dist = euclidean_distances(data)
dist.partition(1, axis=1)
dist_min = dist[:,1]
std = np.std(dist_min)
dim = round(np.exp((std - w[1]) / w[0]))
if dim > 60:
    dim = 60
print(dim)

'''
pred.append(dim)
pred = np.log(pred)
with open('Handpred.csv','w')as csvfile:
    csvfile.write('SetId,LogDim\n')
    for i in range(pred.shape[0]):
        csvfile.write(str(i)+ ',' + str(pred[i]) + '\n')'''
