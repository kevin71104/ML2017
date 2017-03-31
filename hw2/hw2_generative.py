#############################################################
#                   Machine Learning 2017                   #
#                   Hw2 : Classification                    #
#             Generative model (deterministic)              #
#############################################################

import numpy as np
import sys
import csv

def myint(a):
    if a.isdigit() :
        return int(a)
    else:
        return 0

####################### Read X_train ########################
train = []
with open(sys.argv[1],'r') as csvFile:
    for row in csv.reader(csvFile):
        train.append( list( map(myint,row) ) )
# retrieve first row
train  = train[1:]
person = len(train)  # number of person
num_w  = len(train[0])  # number of weight

####################### Read Y_train ########################
label = []
with open(sys.argv[2],'r') as csvFile:
    for row in csv.reader(csvFile):
        label.extend( list( map(myint,row) ) )

#################### Classify Train_data ####################
highpay = []
lesspay = []
for i in range(person):
    if label[i]:
        highpay.append(train[i])
    else:
        lesspay.append(train[i])
#for i in range(10):
#    print(highpay[i])

#################### Calculate mu & sigma ###################
highpay = np.array(highpay)
lesspay = np.array(lesspay)

N_H = highpay.shape[0] # number of high-pay people
N_L = lesspay.shape[0] # number of low-pay people

mu_h = highpay.mean(axis = 0) # 1*M row vector
mu_l = lesspay.mean(axis = 0)

sigma_h = np.zeros((num_w,num_w))
sigma_l = np.zeros((num_w,num_w))
for i in range(N_H):
    sigma_h += np.dot( np.transpose([highpay[i]-mu_h]), [highpay[i]-mu_h] )
for i in range(N_L):
    sigma_l += np.dot( np.transpose([lesspay[i]-mu_l]), [lesspay[i]-mu_l] )

sigma_h = sigma_h / N_H
sigma_l = sigma_l / N_L
sigma = (N_H*sigma_h + N_L*sigma_l)/person

print(sigma.shape)
###################### Calculate w & b ######################
sigma_inv = np.linalg.inv(sigma)


w = np.dot( (mu_h-mu_l), sigma_inv )
b = np.log(float(N_H) / N_L) - 0.5 * np.dot(np.dot([mu_h],sigma_inv),mu_h) \
                             + 0.5 * np.dot(np.dot([mu_l],sigma_inv),mu_l)

####################### Read X_test #########################
test = []
with open(sys.argv[3],'r') as csvFile:
    for row in csv.reader(csvFile):
        test.append( list( map(myint,row) ) )
# retrieve first row
test = np.array(test[1:])

######################## Write Y_test #######################
output = []
z = np.dot(w,test.T) + b
for i in range(z.shape[0]):
    if z[i] > 0 :
        output.append(1)
    else:
        output.append(0)

x = 1
with open(sys.argv[4],'w') as csvFile:
    csvFile.write('id,label')
    for row in output:
        csvFile.write('\n' + str(x) + ',' + str(row))
        x = x+1
