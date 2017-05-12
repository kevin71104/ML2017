################################################################################
#                            Machine Learning 2017                             #
#                               Hw4 : eigenface                                #
################################################################################

from PIL import Image as im
import numpy as np
import matplotlib.pyplot as plt

filepath = './faceExpressionDatabase/'
data = []
for i in range(65,65+10):
    for j in range(10):
        filename = filepath + chr(i) + '0' + str(j) + '.bmp'
        image = im.open(filename)
        arr = np.asarray(image)
        data.append(arr.flatten())
data = np.array(data,'int')
#print(data)
#print(data.shape)
mean = data.mean(axis=0, keepdims=True)
############################### average face ###################################
meanface = mean.reshape(64,64)
figure1 = plt.figure()
ax = figure1.add_subplot(1,1,1)
ax.imshow(meanface,cmap='gray')
ax.set_axis_off()
figure1.suptitle('average face')
figure1.savefig('average_face')
################################ eigenface #####################################
data_ctr = data - mean
u, s, v = np.linalg.svd(data_ctr,full_matrices=False)
eigenface = plt.figure()
for i in range(9):
    face = v[i,:].reshape(64,64)
    ax = eigenface.add_subplot(3,3,i+1)
    ax.set_axis_off()
    ax.imshow(face,cmap='gray')
eigenface.suptitle('eigenface')
eigenface.savefig('eigenface')
