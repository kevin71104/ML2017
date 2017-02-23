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
    # map will return a list
    numbers = map(int, line.split(','))
    y.append(numbers)
f.close()
z = np.mat(x) * np.mat(y)
z = np.array(z)
#print(x)
#print(y)
#print(z)
ans = []
for row in z:
    ans.extend(row)
ans.sort()
f = open('ans_one.txt' , 'w')
for it in ans:
    f.write(str(it)+'\n')
