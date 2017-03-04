import numpy as np
import random as rnd
import sys
import csv

# get train data
train_data = []
with open(sys.argv[1],'r',encoding='big5') as csvFile:
    for row in csv.reader(csvFile):
        if(row[2] == 'PM2.5'):
             train_data.extend( list( map(int,row[3:27]) ) )

# model: y = w0 + wi*xi (i from 1 to 9)
# initialize parameters
w = np.array([4]*10)
gradprev = np.array([1]*10) #0-array
lr = 1
iteration = 10000000
w_his = [w]
num_ex = len(train_data)-9
"""for i in range(3):
    ip = np.array([1]+train_data[i:i+9])
    op = np.array([train_data[i+9]]*10)
    print (ip)
    print (op)"""
# start training
for i in range(iteration):
    print(str(i)+ '\b'*6 +'\n')
    w_grad = np.zeros(10)
    #stochastic: pick only one random example
    n = rnd.randrange(num_ex)
    ip = np.array([1]+train_data[n:n+9])
    temp = (-2)*(train_data[n+9]-np.inner(w,ip))
    grad = np.array([temp]*10)*ip
    gradprev = gradprev + grad**2
    #update parameters
    w = w - lr/np.sqrt(gradprev)*grad
    #store parameters
    w_his.append(w)
# get test data
output = []
with open(sys.argv[2],'r',encoding='big5') as csvFile:
    for row in csv.reader(csvFile):
        if(row[1] == 'PM2.5'):
             test_data = np.array( [1] + list( map(int,row[2:12])) )
             output.append(np.inner(test_data,w_his[-1]))
#write output
x = 0
with open(sys.argv[3],'w') as csvFile:
    csvFile.write('id,value')
    for row in output:
        csvFile.write('\nid_' + str(x) + ',' + str(row))
        x = x+1
