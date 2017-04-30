#############################################################
#                   Machine Learning 2017                   #
#            Hw3 : Image Sentiment Classification           #
#                Convolutional Neural Network               #
#         description : use model to get prediction         #
#        script : python3 test.csv model.h5 ans.csv         #
#############################################################

import pandas as pd
import numpy as np
import sys
from keras.models import load_model
#from keras.utils import np_utils

######################### Read File #########################
with open(sys.argv[1],'r') as csvFile:
    test = pd.read_csv(csvFile)
test_feature = test['feature']
x = []
for i in range(test_feature.shape[0]):
    x.append(test_feature[i].split(' '))
test_feature = np.array(x, dtype=float)
#print(test_feature)
test_feature = test_feature/255
test_feature = test_feature.reshape(test_feature.shape[0],48,48,1)
"""
test_label   = np.array(test.label)
classNum = 7
test_label = np_utils.to_categorical(test_label, classNum)
"""
model = load_model(sys.argv[2])

"""
score = model.evaluate(test_feature, test_label)
print ("\nTrain accuracy:", score[1])
"""
output = model.predict_classes(test_feature,batch_size=100,verbose=1)

x = 0
with open(sys.argv[3],'w') as csvFile:
    csvFile.write('id,label')
    for i in range(len(output)):
        csvFile.write('\n' + str(x) + ',' + str(output[i]))
        x = x+1
