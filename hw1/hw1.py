import numpy as np
import random as rnd
import sys
import csv
"""
# get train data(by month)
train_data = [[] for x in range(12)]
month = 0
count = 0
with open(sys.argv[1],'r',encoding='big5') as csvFile:
    for row in csv.reader(csvFile):
        if(row[2] == 'PM2.5'):
            count = count + 1
            train_data[month].extend( list( map(int,row[3:27]) ) )
            if(count == 20):
                month = month + 1
                count = 0
"""
# get train data
train_data = []
with open(sys.argv[1],'r',encoding='big5') as csvFile:
    for row in csv.reader(csvFile):
        if(row[2] == 'PM2.5'):
            train_data.extend( list( map(int,row[3:27]) ) )
"""
for month in range(len(train_data)):
    print('month'+str(month+1),end = ':')
    for i in range(len(train_data[month])):
        if(i%24 == 0):
            print()
        print(train_data[month][i],end = ' ')
    print()
"""

# model: y = w0 + wi*xi (i from 1 to 9)
# initialize parameters
w = np.array([4]*10)
gradprev = np.array([1]*10) #0-array
lr = 1
lb = 0.1 #regularization coefficient
iteration = 1500000
w_his = [w]
#num_ex = len(train_data[0])-9
num_ex = len(train_data)-9

# start training
for i in range(iteration):
    print(str(i)+ '\b'*7 , end = '')
    w_grad = np.zeros(10)
    #stochastic: pick only one random example
    """by month
    month = rnd.randrange(12)
    n = rnd.randrange(num_ex)
    ip = np.array([1]+train_data[month][n:n+9])
    temp = (-2)*(train_data[month][n+9]-np.inner(w,ip))"""
    n = rnd.randrange(num_ex)
    ip = np.array([1]+train_data[n:n+9])
    temp = (-2)*(train_data[n+9]-np.inner(w,ip))
    grad = np.array([temp]*10)*ip + 2*np.array([0.0]+[lb]*9)*w
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
with open('model'+sys.argv[3],'w') as csvFile:
    for row in range(len(w_his[-1])):
        if(row == 0):
            csvFile.write(str(w_his[-1][row]))
        else:
            csvFile.write(',' + str(w_his[-1][row]))
