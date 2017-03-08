import numpy as np
import random as rnd
import sys
import csv
import math

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

def myfloat(a):
    if isfloat(a):
        return float(a)
    else:
        return 0.0

# get train data
train_data = []
otherdata = [[] for x in range(17)]
with open(sys.argv[1],'r',encoding='big5') as csvFile:
    for row in csv.reader(csvFile):
        if(row[2] == 'PM2.5'):
            train_data.extend( list( map(myfloat,row[3:27]) ) )
        elif(row[2] == 'AMB_TEMP'):
            otherdata[0].extend( list( map(myfloat,row[3:27]) ) )
        elif(row[2] == 'CH4'):
            otherdata[1].extend( list( map(myfloat,row[3:27]) ) )
        elif(row[2] == 'CO'):
            otherdata[2].extend( list( map(myfloat,row[3:27]) ) )
        elif(row[2] == 'NMHC'):
            otherdata[3].extend( list( map(myfloat,row[3:27]) ) )
        elif(row[2] == 'NO'):
            otherdata[4].extend( list( map(myfloat,row[3:27]) ) )
        elif(row[2] == 'NO2'):
            otherdata[5].extend( list( map(myfloat,row[3:27]) ) )
        elif(row[2] == 'NOx'):
            otherdata[6].extend( list( map(myfloat,row[3:27]) ) )
        elif(row[2] == 'O3'):
            otherdata[7].extend( list( map(myfloat,row[3:27]) ) )
        elif(row[2] == 'PM10'):
            otherdata[8].extend( list( map(myfloat,row[3:27]) ) )
        elif(row[2] == 'RAINFALL'):
            otherdata[9].extend( list( map(myfloat,row[3:27]) ) )
        elif(row[2] == 'RH'):
            otherdata[10].extend( list( map(myfloat,row[3:27]) ) )
        elif(row[2] == 'SO2'):
            otherdata[11].extend( list( map(myfloat,row[3:27]) ) )
        elif(row[2] == 'THC'):
            otherdata[12].extend( list( map(myfloat,row[3:27]) ) )
        elif(row[2] == 'WD_HR'):
            otherdata[13].extend( list( map(myfloat,row[3:27]) ) )
        elif(row[2] == 'WIND_DIREC'):
            otherdata[14].extend( list( map(myfloat,row[3:27]) ) )
        elif(row[2] == 'WIND_SPEED'):
            otherdata[15].extend( list( map(myfloat,row[3:27]) ) )
        elif(row[2] == 'WS_HR'):
            otherdata[16].extend( list( map(myfloat,row[3:27]) ) )
otherdata = np.array(otherdata)
otherdata = np.transpose(otherdata)
# model: y = w0 + wi*xi (i from 1 to 9) +w10*x8^2 + w11*x9^2
# initialize parameters
w = np.array([(0.5)]*12)
gradprev = np.array([0]*12) #0-array
lr = 1
lb = 0.0001 #regularization coefficient
lb2 = 0.00001
iteration = 5000000
w_his = [w]
#num_ex = len(train_data[0])-9
num_ex = len(train_data)-9


# start training
for i in range(iteration):
    print(str(i)+ '\b'*7 , end = '')
    w_grad = np.zeros(12)
    #stochastic: pick only one random example
    """by month
    month = rnd.randrange(12)
    n = rnd.randrange(num_ex)
    ip = np.array([1]+train_data[month][n:n+9])
    temp = (-2)*(train_data[month][n+9]-np.inner(w,ip))"""
    n = rnd.randrange(num_ex)
    """while n%480 > 471:
        print(str(i)+' '+str(n%480))
        n = rnd.randrange(num_ex)"""
    ip = np.array([1]+train_data[n:n+9]+[train_data[n+7]**2]+[train_data[n+8]**2])
    temp = (-2)*(train_data[n+9]-np.inner(w,ip))
    grad = np.array([temp]*12)*ip + 2*np.array([0.0]+[lb]*9+[lb2]*2)*w
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
             test_data = np.array( [1] + list( map(int,row[2:11])) +[int(row[9])**2]+[int(row[10])**2])
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
