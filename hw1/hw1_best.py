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
prevdata = []
otherdata = [[] for x in range(17)]
with open(sys.argv[1],'r',encoding='big5') as csvFile:
    for row in csv.reader(csvFile):
        if(row[2] == 'PM2.5'):
            prevdata.extend( list( map(myfloat,row[3:27]) ) )
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
"""
x = 0
for row in otherdata[9]:
    x = x+1
    print (row, end = ' ')
    if (x % 24 == 0): print('\n')
"""
# model: y = b + wi*xi (i from 0 to 25)
# prev 9 hours + other 17 parameters
# initialize parameters
b = 4.0
w = np.array([4.0]*26)
lb = 10 #regularization coefficient
bprev = 1.0
gradprev = np.array([1.0]*26)
lr = 1
iteration = 1500000
b_his = [b]
w_his = [w]
num_ex = len(prevdata)-9


# start training
for i in range(iteration):
    print(str(i)+ '\b'*8,end = '')
    b_grad = 0.0
    w_grad = np.zeros(26)
    #stochastic: pick only one random example
    n = rnd.randrange(num_ex)
    ip = np.append(np.array(prevdata[n:n+9]) , otherdata[n+9])
    #print(n)
    #print(ip)
    temp = (-2)*(prevdata[n+9]-b-np.inner(w,ip))
    grad = np.array([temp]*26)*ip + 2*lb*w #lb : regularization
    bprev = bprev + (temp+2*lb*b)**2
    gradprev = gradprev + grad**2
    #update parameters
    b = b - lr/math.sqrt(bprev)*temp
    w = w - lr/np.sqrt(gradprev)*grad
    #store parameters
    w_his.append(w)
    b_his.append(b)
# get test data
other = []
output = []
i = 0
with open(sys.argv[2],'r',encoding='big5') as csvFile:
    for row in csv.reader(csvFile):
        i = i+1
        if(row[1] == 'PM2.5'):
            prev = list( map(myfloat,row[2:11]))
            #print(prev)
        else:
            other.append(myfloat(row[10]))
        if(i%18 == 0):
            test_data = np.array(prev + other)
            output.append( b_his[-1] + np.inner(test_data,w_his[-1]) )
            other = []

#write output
x = 0
with open(sys.argv[3],'w') as csvFile:
    csvFile.write('id,value')
    for row in output:
        csvFile.write('\nid_' + str(x) + ',' + str(row))
        x = x+1
with open('model'+sys.argv[3],'w') as csvFile:
    csvFile.write(str(b_his[-1]))
    for row in w_his[-1]:
        csvFile.write(' ' + str(row))
