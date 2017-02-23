import numpy as np
import sys
f = open(sys.argv[1] , 'r+')
x = []
for line in f:
    numbers = map(int, line.split(','))
    x.append(numbers)
f.close()
f = open(sys.argv[2] , 'r')
y = []
for line in f:
    numbers = map(int, line.split(','))
    y.append(numbers)
f.close()
#print(y)
z = np.mat(x) * np.mat(y)
z = np.array(z)
z.sort()
f = open('ans_one.txt' , 'w')
for row in z:
    for it in row:
        f.write(str(it)+'\n')
