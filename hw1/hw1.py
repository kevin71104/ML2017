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
        if ( float(a) > 0):
            return float(a)
        else:
            return 0.0
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

mean = np.mean(train_data);
var  = np.var(train_data);

# model: y = w0 + wi*xi (i from 1 to 9) +w10*x8^2 + w11*x9^2
# initialize parameters
#w  = np.array([0.001]*8 + [0.01, 1.1, 0.001, 0.003])
#lr = np.array([0.05]*8 + [0.0005, 0.5, 0.1, 0.00001])
w  = np.array([1.0]*8 + [0.01 , 0.1, 0, 0.004])
lr = np.array([1]*8   + [0.0005, 0.1, 0,   0.00002])
gradprev = np.array([0.0]*12)
lb = 0.001 #regularization coefficient
lb2 = 0.000001
iteration = 3000000
w_his = [w]
num_ex = len(train_data)-9

# start training
success   = 0
traintime = 0
while(1):
#for i in range(iteration):
    print(str(traintime)+ '\b'*7 , end = '')
    w_grad = np.zeros(12)
    #stochastic: pick only one random example
    n = rnd.randrange(num_ex)
    while( n % 480 > 470):
        n = rnd.randrange(num_ex)
    ip = np.array([1]+train_data[n:n+9]+[train_data[n+7]**2]+[train_data[n+8]**2])
    temp = (train_data[n+9]-np.inner(w,ip))
    if(temp < 0.0003 and temp > -0.0003):
        success = success + 1
        print('success: ' + str(success) + 'temp: ' + str(temp))
        if(success > 100 or traintime > iteration):
            #if(temp < 0.0001 and temp > -0.0001):
                print('last success: ' + str(temp))
                break
    grad = np.array([temp*(-2)]*12)*ip + 2*np.array([0.0]+[lb]*8+[lb2]*3)*w
    #grad = np.array([temp]*12)*ip
    gradprev = gradprev + grad**2
    #update parameters
    w = w - lr/np.sqrt(gradprev)*grad
    #store parameters
    w_his.append(w)
    traintime = traintime + 1
print('\nsuccess: ' + str(success))
print('iteration: ' + str(traintime))

# get test data
output = []
with open(sys.argv[2],'r',encoding='big5') as csvFile:
    for row in csv.reader(csvFile):
        if(row[1] == 'PM2.5'):
             test_data = np.array( [1] + list( map(myfloat,row[2:11])) +[myfloat(row[9])**2]+[myfloat(row[10])**2])
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
        csvFile.write(str(row)+',' + str(w_his[-1][row])+'\n')
    csvFile.write('\n\nw_history:\n')
    """for i in range(5):
        csvFile.write('number'+str(i)+'\n')
        for row in range(len(w_his[-1])):
            csvFile.write(str(w_his[i][row])+', ')
        csvFile.write('\n')
    for i in range(5):
        csvFile.write('number'+str(len(w_his)-6+i)+'\n')
        for row in range(len(w_his[-1])):
            csvFile.write(str(w_his[len(w_his)-6+i][row])+', ')
        csvFile.write('\n')"""
    for i in range(len(w_his)):
        if((i%5000) == 0 ):
            csvFile.write(str(i)+':')
            csvFile.write(str(w_his[i][0])+ ' ')
            csvFile.write(str(w_his[i][8:12]))
            csvFile.write('\n')
