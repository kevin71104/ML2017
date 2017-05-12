################################################################################
#                            Machine Learning 2017                             #
#                               Hw4 : eigenface                                #
################################################################################

from PIL import Image as im
import numpy as np
from numpy.linalg import norm
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
figure1.suptitle('Average face')
figure1.savefig('average_face.png')
################################ eigenface #####################################
data_ctr = data - mean
u, s, v = np.linalg.svd(data_ctr,full_matrices=False)
eigenface = plt.figure()
for i in range(9):
    face = v[i,:].reshape(64,64)
    subface = eigenface.add_subplot(3,3,i+1)
    subface.set_axis_off()
    subface.imshow(face,cmap='gray')
eigenface.suptitle('Eigenfaces')
eigenface.savefig('eigenface.png')
################################### reconstruct ################################
'''reducematrix = v[0:5,:]
coefficient = np.dot(data_ctr,np.transpose(reducematrix)) # c_i,j = (data_i-mu) dot w_j
recover = plt.figure(figsize=(20, 20))
origin = plt.figure(figsize=(20, 20))
for i in range(100):
    print(i)
    recoverdata = np.sum(coefficient[i,:].reshape(5,1) * reducematrix,axis=0) + mean
    recoverface = recoverdata.reshape(64,64)
    subface = recover.add_subplot(10,10,i+1)
    subface.set_axis_off()
    subface.imshow(recoverface,cmap='gray')

    originface = data[i,:].reshape(64,64)
    subface2 = origin.add_subplot(10,10,i+1)
    subface2.set_axis_off()
    subface2.imshow(originface,cmap='gray')

recover.suptitle('Reconstruct',fontsize=36)
recover.savefig('reconstruct.png')
origin.suptitle('Original',fontsize=36)
origin.savefig('original.png')'''

for k in range(1,v.shape[0]+1):
    RMSE = 0
    reducematrix = v[:k,:]
    coefficient = np.dot(data_ctr,np.transpose(reducematrix))
    for i in range(100):
        recoverdata = np.sum(coefficient[i,:].reshape(k,1) * reducematrix,axis=0) + mean
        error = np.square(norm(data[i,:]-recoverdata))
        RMSE = RMSE + error
    RMSE = np.sqrt((RMSE / data.shape[0]) / data.shape[1]) / 255
    print('RMSE of reduction dimension({}): {}'.format(k,RMSE))
    if RMSE < 0.01:
        print('At reduction dimension({}), RMSE < 1%'.format(k))
        break
